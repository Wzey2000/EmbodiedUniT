from collections import deque
import torch
# this wrapper comes after vectorenv
from env_utils.env_wrapper.graph import Graph
from env_utils.env_wrapper.env_graph_wrapper import GraphWrapper


class PretrainWrapper(GraphWrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,exp_config, batch_size):
        self.exp_config = exp_config
        self.is_vector_env = True
        self.num_envs = batch_size
        self.B = self.num_envs
        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.scene_data = exp_config.scene_data
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.visual_encoder_type = getattr(exp_config, 'visual_encoder_type', 'unsupervised')
        self.visual_encoder = self.load_visual_encoder(self.visual_encoder_type, self.input_shape, self.feature_dim).to(self.torch_device)

        self.graph = Graph(exp_config, self.B, self.torch_device)
        self.th = getattr(exp_config, 'GRAPH_TH', 0.75) # default: 0.75
        self.num_agents = exp_config.NUM_AGENTS
        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY
        self.localize_mode = 'predict'

        # for forgetting mechanism
        self.forget = exp_config.memory.FORGET
        self.forgetting_recorder = None
        self.forget_node_indices = None
        self.reset_all_memory()

    def step(self, batch):
        demo_rgb_t, demo_depth_t, positions_t, target_img, t, mask = batch
        obs_batch = {}
        obs_batch['step'] = t
        obs_batch['target_goal'] = target_img
        obs_batch['panoramic_rgb'] = demo_rgb_t
        obs_batch['panoramic_depth'] = demo_depth_t
        obs_batch['position'] = positions_t
        curr_vis_embedding = self.embed_obs(obs_batch)
        self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), t, mask)
        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        obs_batch['curr_embedding'] = curr_vis_embedding
        return obs_batch

    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,exp_config, batch_size):
        #super().__init__(None, exp_config)
        self.is_vector_env = True
        self.num_envs = batch_size
        self.B = self.num_envs
        self.feature_dim = exp_config.features.visual_feature_dim
        self.device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.rgb_memory = deque(maxlen=exp_config.memory.memory_size)
        self.depth_memory = deque(maxlen=exp_config.memory.memory_size)
        self.graph = Graph(exp_config, self.B, self.device)
        self.graph.reset(self.B)
        self.memory_size = exp_config.memory.memory_size
        self.embedding_idxs = torch.tensor(data=[[0,1,2,3]], dtype=int).repeat(self.B,1)
        self.mask = torch.ones(size=(self.B, self.memory_size), dtype=bool)
        self.step_cnt = 0

        self.goal_encoder = self.load_visual_encoder(self.feature_dim).to(self.device) # Custom ResNet18
        self.goal_encoder.eval()

        #self.scene_data = exp_config.scene_data

        self.reset_all_memory()

    def reset_all_memory(self, B=None):
        self.rgb_memory.clear()
        self.depth_memory.clear()
        self.embedding_idxs = torch.tensor(data=[[0,1,2,3]], dtype=int).repeat(self.B,1)
        self.step_cnt = 0
    
    def step(self, batch):
        demo_rgb_t, demo_depth_t, positions_t, target_img, t, mask = batch
        obs_batch = {}
        obs_batch['step'] = t
        obs_batch['target_goal'] = target_img
        obs_batch['panoramic_rgb'] = demo_rgb_t
        obs_batch['panoramic_depth'] = demo_depth_t

        self.rgb_memory.append(demo_rgb_t)
        self.depth_memory.append(demo_depth_t)

        depth_history = list(self.depth_memory)
        depth_history.extend([depth_history[-1]] * (self.depth_memory.maxlen - len(depth_history))) 
        obs_batch['panoramic_depth_history'] = torch.stack(depth_history)
        
        rgb_history = list(self.rgb_memory)
        rgb_history.extend([rgb_history[-1]] * (self.rgb_memory.maxlen - len(rgb_history)))
        obs_batch['panoramic_rgb_history'] = torch.stack(rgb_history)
        
        obs_batch['global_mask'] = self.mask

        obs_batch['position'] = positions_t
        obs_batch['global_memory'] = self.embedding_idxs
        obs_batch['goal_embedding'] = self.embed_target(obs_batch)
        #curr_vis_embedding = self.embed_obs(obs_batch)
        #global_memory_dict = self.get_global_memory()
        #obs_batch['curr_embedding'] = curr_vis_embedding
        return obs_batch