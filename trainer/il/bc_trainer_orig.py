import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import imageio
import cv2
import copy

class BC_trainer:#(nn.Module):
    def __init__(self, cfg, agent, optim, device):
        super().__init__()
        self.agent = agent
        #self.env_wrapper = BCWrapper(cfg, observation_space)
        self.feature_dim = cfg.features.visual_feature_dim
        self.device = device
        self.optim = optim
        self.config = cfg
        self.env_setup_done = False
        self.num_recurrent_layers = cfg.STATE_ENCODER.num_recurrent_layers
        self.output_size = cfg.STATE_ENCODER.hidden_size
        self.max_grad_norm = cfg.BC.max_grad_norm
        self.use_iw = cfg.BC.USE_IW
        self.use_aux_tasks = cfg.USE_AUXILIARY_INFO

        self.optimizer_to_GPU()
    
    def optimizer_to_GPU(self):
        for param in self.optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)
    
    def save(self,file_name=None, env_global_node=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.module.state_dict()
            save_dict['optimizer'] = self.optim.state_dict()
            if env_global_node is not None:
                save_dict['env_global_node'] = env_global_node
            torch.save(save_dict, file_name)
    
    def train(self, batch, env_global_node, train=True):
        demo_rgb, demo_depth, demo_act, positions, rotations, targets, target_img, scene, start_pose, aux_info = batch
        demo_rgb, demo_depth, demo_act = demo_rgb.to(self.device), demo_depth.to(self.device), demo_act.to(self.device)
        target_img, positions, rotations = target_img.to(self.device), positions.float().to(self.device), rotations.float().to(self.device)
        positions -= positions[:,0:1]
        rotations -= rotations[:,0:1]

        aux_info = {
                    # 'have_been': aux_info['have_been'].to(self.device),
                    # 'distance': aux_info['distance'].to(self.device),
                    'IW': aux_info['IW'].to(self.device)}
        self.B = demo_act.shape[0]
        # self.env_wrapper.B = demo_act.shape[0]
        # self.env_wrapper.reset_all_memory(self.B)
        lengths = (demo_act > -10).sum(dim=1)

        T = lengths.max().item()
        hidden_states = torch.zeros(self.num_recurrent_layers, self.B, self.output_size).to(self.device)

        actions = torch.zeros([self.B], device=self.device)
        results = {'imgs': [], 'curr_node': [], 'node_list':[], 'actions': [], 'gt_actions': [], 'target': [], 'scene':scene[0], 'A': [], 'position': [],
                   'have_been': [], 'distance': [], 'pred_have_been': [], 'pred_distance': []}
        losses = []
        # span_losses = []
        # aux_losses1 = []
        # aux_losses2 = []
        for t in range(T):
            masks = lengths > t
            if t == 0: masks[:] = False
            target_goal = target_img[torch.arange(0,self.B).long(), targets[:,t].long()]

            obs_t = {}
            obs_t['step'] = torch.ones(self.B, device=self.device)*t
            #obs_t['target_goal'] = target_goal
            obs_t['rgb'] = torch.cat([demo_rgb[:,t], target_goal], dim=0)
            obs_t['depth'] = demo_depth[:,t]
            obs_t['position'] = positions[:,t]
            obs_t['rotation'] = rotations[:,t]
            # if t < lengths[0]:
            #     results['imgs'].append(demo_rgb[0,t].cpu().numpy())
            #     results['target'].append(target_goal[0].cpu().numpy())
            #     results['position'].append(positions[0,t].cpu().numpy())
            #     results['have_been'].append(aux_info['have_been'][0,t].cpu().numpy())
            #     results['distance'].append(aux_info['distance'][0,t].cpu().numpy())
            #     results['node_list'].append(copy.deepcopy(self.env_wrapper.graph.node_position_list[0]))
            #     results['curr_node'].append(self.env_wrapper.graph.last_localized_node_idx[0].cpu().numpy())

            gt_act = copy.deepcopy(demo_act[:, t])
            if -100 in actions:
                b = torch.where(actions==-100)
                actions[b] = 0

            (
                pred_act,
                actions_logits,
                hidden_states,
            ) = self.agent.module.act(
                obs_t,
                hidden_states,
                actions.view(self.B,1),
                masks.unsqueeze(1),
            )
                    
            if not (gt_act == -100).all():
                
                loss = F.cross_entropy(actions_logits.view(-1,actions_logits.shape[1]),gt_act.long().view(-1), reduction='none')#, weight=action_weight)
                if self.use_iw:
                    loss = (loss * aux_info['IW'][:,t]).mean()
                else:
                    loss = loss.mean()
                # pred1, pred2 = preds
                valid_indices = gt_act.long() != -100
                # aux_loss1 = F.binary_cross_entropy_with_logits(pred1[valid_indices].view(-1), aux_info['have_been'][valid_indices,t].float().reshape(-1))
                # aux_loss2 = F.mse_loss(F.sigmoid(pred2)[valid_indices].view(-1), aux_info['distance'][valid_indices,t].float().reshape(-1))

                losses.append(loss)
                # aux_losses1.append(aux_loss1)
                # aux_losses2.append(aux_loss2)
    
                results['actions'].append(pred_act[0].detach().cpu().numpy())
                results['gt_actions'].append(int(gt_act[0].detach().cpu().numpy()))

            else:
                results['actions'].append(-1)
                results['gt_actions'].append(-1)

            
            # results['pred_have_been'].append(F.sigmoid(pred1)[0].detach().cpu().numpy())
            # results['pred_distance'].append(F.sigmoid(pred2)[0].detach().cpu().numpy())
            actions = demo_act[:,t].contiguous()

        action_loss = torch.stack(losses).mean()

        # aux_loss1 =torch.stack(aux_losses1).mean()
        # aux_loss2 = torch.stack(aux_losses2).mean()
        total_loss = action_loss# + aux_loss1 + aux_loss2
        if train:
            self.optim.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
            self.optim.step()

        loss_dict = {}
        loss_dict['loss'] = action_loss.item()
        # loss_dict['aux_loss1'] = aux_loss1.item()
        # loss_dict['aux_loss2'] = aux_loss2.item()
        return results, loss_dict, None

    def visualize(self, result_dict, file_name, mode='train'):
        if mode == 'train':
            imgs = result_dict['imgs']
            target = result_dict['target']
            acts, gt_acts = result_dict['actions'], result_dict['gt_actions']
            if 'node_list' in result_dict:
                node_list, curr_node, position = result_dict['node_list'], result_dict['curr_node'], result_dict['position']
            if 'have_been' in result_dict:
                have_been, distance = result_dict['have_been'], result_dict['distance']
                pred_have_been, pred_distance = result_dict['pred_have_been'], result_dict['pred_distance']

            writer = imageio.get_writer(file_name + '.mp4', fps=15)
            for t in range(len(imgs)):
                view_im = imgs[t]
                target_im = target[t][:,:,:3] * 255
                view_im = np.concatenate([view_im, target_im],0).astype(np.uint8)
                view_im = cv2.resize(view_im,(256,128))
                cv2.putText(view_im, "t: %03d"%t + " act {} gt_act {}".format(acts[t], gt_acts[t]), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
                if 'node_list' in result_dict and len(result_dict['node_list']) > 0:
                    node_idx = np.linalg.norm(np.array(node_list[t]) - np.array(position[t]).reshape(1,-1), axis=1).argmin()
                    cv2.putText(view_im, "num_node : %d, curr_node: %d , gt_node:%d" % (len(node_list[t]), curr_node[t], node_idx), (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view_im, "have_been: %.3f / %d     dist: %.3f/%.3f"%(pred_have_been[t], have_been[t], pred_distance[t], distance[t]),
                            (20, 40 + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                writer.append_data(view_im)
            writer.close()
        else:
            imgs = result_dict['imgs']
            writer = imageio.get_writer(file_name+'.mp4')
            w,h = imgs[-1].shape[0],imgs[-1].shape[1]
            for t in range(len(imgs)):
                view_im = cv2.resize(imgs[t],(h,w))
                writer.append_data(view_im)
            writer.close()