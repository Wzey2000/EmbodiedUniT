import torch
from torchvision import transforms
from utils.encoder_loaders import load_PCL_encoder, load_CRL_encoder

# this wrapper comes after vectorenv
from env_utils.env_wrapper.graph import Graph
from env_utils.env_wrapper.env_graph_wrapper import GraphWrapper

class BCWrapper(GraphWrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config, observation_space):
        self.config = config
        self.is_vector_env = True
        self.num_envs = config.BC.batch_size
        self.B = self.num_envs
        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.scene_data = config.scene_data
        self.torch = config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.visual_encoder_type = getattr(config, 'visual_encoder_type', 'unsupervised')
        
        self.pretrained_type = config.memory.pretrained_type
        if self.pretrained_type == 'PCL':
            self.visual_encoder = load_PCL_encoder(self.feature_dim)
        elif self.pretrained_type == 'CRL':
            if 'depth' in observation_space.spaces:
                observation_space.spaces.pop('depth')
            elif 'panoramic_depth' in observation_space.spaces:
                observation_space.spaces.pop('panoramic_depth')
            
            self.trans = transforms.Compose([
                transforms.Resize((256,256))
            ])
            self.visual_encoder = load_CRL_encoder(observation_space)
        
        self.visual_encoder.eval()
        self.visual_encoder.to(self.torch_device)

        for p in self.visual_encoder.parameters():
            p.requires_grad = False
        self.graph = Graph(config, self.B, self.torch_device)
        self.th = 0.75
        self.num_agents = config.NUM_AGENTS
        self.need_goal_embedding = 'wo_Fvis' in config.POLICY
        self.localize_mode = 'predict'

        # for forgetting mechanism
        self.forget = config.memory.FORGET
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
