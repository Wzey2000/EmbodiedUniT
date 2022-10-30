from runner.base_runner import BaseRunner
import torch
import time
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from env_utils.env_wrapper.base_graph_wrapper import BaseGraphWrapper
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import quaternion as q
from model.policy import *
from utils.utils import get_model_summary
from roma import unitquat_to_rotvec

class CNNRNNRunner(nn.Module):
    def __init__(self, config, env_global_node=None, return_features=False):
        super().__init__()
        self.config = config
        observation_space = SpaceDict({
            'rgb': Box(low=0, high=256, shape=(240, 320, 3), dtype=np.float32),
            'depth': Box(low=0, high=256, shape=(240, 320, 1), dtype=np.float32),
            'target_goal': Box(low=0, high=256, shape=(240, 320, 3), dtype=np.float32),
            'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
            'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "gps": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(config.TASK_CONFIG.TASK.GPS_SENSOR.DIMENSIONALITY,),
                dtype=np.float32),
        })
        action_space = Discrete(config.ACTION_DIM)
        print(config.POLICY, 'using ', eval(config.POLICY))

        agent = eval(config.POLICY)(
            observation_space=observation_space,
            action_space=action_space,
            no_critic=True,
            normalize_visual_inputs=True,
            config=config
        )
        self.agent = agent
        self.torch_device = (
            torch.device("cuda:"+str(config.TORCH_GPU_ID[0]))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.return_features = return_features
        self.need_env_wrapper = True
        self.num_agents = 1

        # settings of global node
        with_env_global_node = config.GCN.WITH_ENV_GLOBAL_NODE
        respawn_env_global_node = config.GCN.RESPAWN_GLOBAL_NODE
        randominit_env_global_node = config.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        global_node_featdim = config.features.visual_feature_dim

        self._env_global_node = env_global_node # the original copy of env global node

        if with_env_global_node:
            if respawn_env_global_node:
                self._env_global_node = torch.randn(1, global_node_featdim) if randominit_env_global_node else torch.zeros(1, global_node_featdim)
            self.env_global_node = self._env_global_node.unsqueeze(0)

        self.use_gpscompass = config.USE_GPS_COMPASS
        self.initial_pose = None
        self.initial_rot = 0
        #self.calc_GFLOPs()
     
    def reset(self, obs):
        self.B = 1
        self.hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B,
                                         self.agent.net.output_size).to(self.torch_device)
        self.env_global_node = self._env_global_node.unsqueeze(0).to(self.torch_device) if self._env_global_node is not None else None
        self.actions = torch.zeros([self.B], device=self.torch_device)
        self.time_t = 0

        if self.use_gpscompass: 
            self.initial_pose = obs['position'][0]
            quat = obs['rotation'].cpu().numpy()
            temp = q.as_euler_angles(q.from_float_array(quat)).astype(np.float32)

            self.initial_rot = torch.tensor(temp[0,1], device=self.torch_device)


    def step(self, obs, reward, done, info, env=None):
        new_obs = {}
        for k, v in obs.items():
            if v is None:
                new_obs[k] = v
            elif isinstance(v, np.ndarray):
                new_obs[k] = torch.from_numpy(v).float().to(self.torch_device).unsqueeze(0)
            elif not isinstance(v, torch.Tensor) and not isinstance(v, set):
                new_obs[k] = torch.tensor(v).float().to(self.torch_device).unsqueeze(0)
            else:
                new_obs[k] = v
        
        if self.use_gpscompass: # Relative to the inital pose
            new_obs['rotation'] = torch.from_numpy((q.as_euler_angles(q.from_float_array(new_obs['rotation'].cpu().numpy())).astype(np.float32))[:,1]).to(self.torch_device)
            new_obs['position']  = torch.cat(
                [-(new_obs['position'][:,2:] - self.initial_pose[2]), new_obs['position'][:,0:1] - self.initial_pose[0]],
                dim=-1)
            new_obs['rotation'] -= self.initial_rot
        
        new_obs['rgb'] = torch.cat([new_obs['rgb'], new_obs['target_goal']], dim=0)
        obs = new_obs

        t = time.time()
        (
            actions,
            actions_logits,
            hidden_states,
        ) = self.agent.act(
            obs,
            self.hidden_states,
            self.actions,
            torch.ones(self.B, device=self.torch_device).unsqueeze(1) * (1-done),
        )
        decision_time = time.time() - t

        # pred1, pred2 = preds

        # if pred1 is not None:
        #     have_been = F.sigmoid(pred1[0])
        #     have_been_str = 'have_been: '
        #     have_been_str += '%.3f '%(have_been.item())
        # else: have_been_str = ''
        # if pred2 is not None:
        #     pred_target_distance = F.sigmoid(pred2[0])
        #     pred_dist_str = 'pred_dist: '
        #     pred_dist_str += '%.3f '%(pred_target_distance.item())
        # else: pred_dist_str = ''

        # log_str = have_been_str + ' ' + pred_dist_str
        # self.env.log_info(log_type='str', info=log_str)
        self.hidden_states = hidden_states
        #self.env_global_node = new_env_global_node
        self.actions = actions # store the previous action
        self.time_t += 1

        return self.actions.item(), None, decision_time

    def visualize(self, env_img):
        return NotImplementedError

    def setup_env(self):
        return

    def wrap_env(self, env, config):
        self.env = BaseGraphWrapper(env, config)
        return self.env

    def get_mean_dist_btw_nodes(self):
        # assume batch size is 1
        dists = []
        for node_idx in range(len(self.node_list[0])):
            neighbors = torch.where(self.A[0, node_idx])[0]
            curr_node_position = self.node_list[0][node_idx].cpu().numpy()
            curr_dists = []
            for neighbor in neighbors:
                if neighbor <= node_idx: continue
                dist = self.env.habitat_env._sim.geodesic_distance(curr_node_position,
                                                                   self.node_list[0][neighbor].cpu().numpy())
                if np.isnan(dist):
                    dist = np.linalg.norm(curr_node_position - self.node_list[0][neighbor].cpu().numpy())
                curr_dists.append(dist)
            if len(curr_dists) > 0:
                dists.append(min(curr_dists))
        # print('A sum' , self.A.sum(), 'num dists', len(dists))
        return dists

    def load(self, state_dict):
        self.agent.load_state_dict(state_dict)
    
    def save(self, file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)
