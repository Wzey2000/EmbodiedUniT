from gym.wrappers.monitor import Wrapper
from gym.spaces.box import Box
import torch
import numpy as np
from custom_habitat.utils.ob_utils import batch_obs

import os
# this wrapper comes after vectorenv
from custom_habitat.core.vector_env import VectorEnv
from env_utils.env_wrapper.graph import Graph


# To learn the functionalities of Wrapper class, see https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
class BaseGraphWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,envs, config):
        self.config = config
        self.envs = envs # SearchEnv or MultiSearchEnv inherited from RLEnv inherited from gym.Env
        self.env = self.envs
        if isinstance(envs,VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1

        self.B = self.num_envs
        self.scene_data = config.scene_data
        self.feature_dim = 512
        self.torch = config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(config.TORCH_GPU_ID[0]) if torch.cuda.device_count() > 0 else 'cpu'

        self.scene_data = config.scene_data

        self.th = 0.0
        self.graph = Graph(config, self.B, self.torch_device)
        self.need_goal_embedding = 'wo_Fvis' in config.POLICY
 
        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.feature_dim),
                                          dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32),
                     'global_A': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.graph.M), dtype=np.float32),
                     'global_time': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32)
                     }
                )
                if self.need_goal_embedding:
                    obs_space.spaces.update(
                        {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                    )                     
        self.num_agents = config.NUM_AGENTS
        self.dummy_feat = torch.zeros(size=(self.B, 512), device=self.torch_device)
        self.reset_all_memory()
    

    def reset_all_memory(self, B=None):
        self.graph.reset(B)

            
    def is_close(self, embed_a, embed_b, return_prob=False):
        with torch.no_grad():
            logits = torch.matmul(embed_a.unsqueeze(1), embed_b.unsqueeze(2)).squeeze(2).squeeze(1)
            close = (logits > self.th).detach().cpu()
        if return_prob: return close, logits
        else: return close

    # assume memory index == node index
    def localize(self, new_embedding, position, time, done_list):
        # The position is only used for visualizations.
        # done_list contains all Trues when navigation starts

        done = np.where(done_list)[0] # 一个参数np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)

        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position)

        close = self.is_close(self.graph.last_localized_node_embedding, new_embedding, return_prob=False)
        found = torch.tensor(done_list) + close # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        
        # 图更新条件一：如果当前时刻智能体和上一时刻位置相同，则更新所处结点的视觉特征
        # Only time infos are updated as no embeddings are provided
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])
        
        # 以下是图更新条件二和三
        # first prepare all available nodes as 0s, and secondly set visited nodes as 1s
        # graph_mask中将每个导航进程的所有现存地图结点都用1来表示
        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0

        check_list[found_batch_indices] = 1.0

        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop : k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b,self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1-check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                batch_close, batch_prob = self.is_close(neighbor_embedding, batch_new_embedding, return_prob=True)
                close = batch_close.split(num_neighbors)
                prob = batch_prob.split(num_neighbors)

                for ii in range(len(close)):
                    is_close = torch.where(close[ii] == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()]
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()]
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node

                        # 图更新条件二： If the current location and the last localized node are different, a new edge between vi and vn is added.
                        # The embedding of vi is replaced with the current feature
                        if found_node != self.graph.last_localized_node_idx[b]:
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])

                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        # 图更新条件三：If the current location cannot be localized in the VGM, a new node vNt+1 with embedding et and an edge between the new node and vn are added to the VGM.
        batch_indices_to_add_new_node = torch.where(to_add)[0]

        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b) # 图结点从0开始编号
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    def update_graph(self):
        if self.is_vector_env:
            args_list = [{'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_graph']*self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b],'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]}}
            self.envs.update_graph(**input_args)

    def update_obs(self, obs_batch, global_memory_dict):
        # add memory to obs
        obs_batch.update(global_memory_dict)
        obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        if 'distance' in obs_batch.keys():
            obs_batch['distance'] = obs_batch['distance']#.unsqueeze(1)

        return obs_batch

    def step(self, actions):
        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]

        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]
        obs_batch = batch_obs(obs_list, device=self.torch_device)

        self.localize(self.dummy_feat, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], done_list)

        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        self.update_graph()

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]
        obs_batch = batch_obs(obs_list, device=self.torch_device)

        # posiitons are obtained by calling habitat_env.sim.get_agent_state().position
        self.localize(self.dummy_feat, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], [True]*self.B)

        global_memory_dict = self.get_global_memory()

        # obs_batch contains following keys:
        # ['rgb_0'~'rgb_11', 'depth_0'~'depth_11', 'panoramic_rgb', 'panoramic_depth',
        # 'target_goal', 'episode_id', 'step', 'position', 'rotation', 'target_pose', 'distance', 'have_been',
        # 'target_dist_score', 'global_memory', 'global_act_memory', 'global_mask', 'global_A', 'global_time', 'forget_mask', 'localized_idx']
        # NOTE: if multiple goals are set, target_goal will have a shape [B, num_goals, 64, 252, 4]
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        
        self.update_graph()

        return obs_batch

    def get_global_memory(self, mode='feature'):
        self.graph
        global_memory_dict = {
            'global_memory': self.graph.graph_memory,
            'global_act_memory': self.graph.graph_act_memory,
            'global_mask': self.graph.graph_mask,
            'global_A': self.graph.A,
            'global_time': self.graph.graph_time,
        }
        return global_memory_dict

    def call(self, aa, bb):
        return self.envs.call(aa,bb)
    def log_info(self,log_type='str', info=None):
        return self.envs.log_info(log_type, info)

    @property
    def habitat_env(self): return self.envs.habitat_env
    @property
    def noise(self): return self.envs.noise
    @property
    def current_episode(self):
        if self.is_vector_env: return self.envs.current_episodes
        else: return self.envs.current_episode
    @property
    def current_episodes(self):
        return self.envs.current_episodes
