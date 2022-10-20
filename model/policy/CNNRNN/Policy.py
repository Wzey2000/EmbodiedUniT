from opcode import hasconst
import torch
import torch.nn as nn
import numpy as np
from habitat import Config, logger
from gym import Space
from gym.spaces import Dict, Box
from model.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.utils.common import CategoricalNet
from model.resnet import resnet
from model.resnet.resnet import ResNetEncoder

from custom_habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)

from custom_habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor,
    task_cat2mpcat40,
    mapping_mpcat40_to_goal21
)

from model.resnet.resnet_encoders import (
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
)
class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

class CNNRNNPolicy(nn.Module):
    def __init__(
            self,
            observation_space, # a SpaceDict instace. See line 35 in train_bc.py
            action_space,
            no_critic=False,
            normalize_visual_inputs=True,
            config=None
    ):
        super().__init__()
        self.net = CNNRNNNet(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            normalize_visual_inputs=normalize_visual_inputs
            
        )
        self.dim_actions = action_space.n # action_space = Discrete(config.ACTION_DIM)
        self.no_critic = no_critic
        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

        if not self.no_critic:
            self.critic = CriticHead(self.net.output_size) # a single layer FC

    def act(
            self,
            observations, # obs are generated by calling env_wrapper.step() in line 59, bc_trainer.py
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
            return_features=False,
            mask_stop=False
    ):
    # observations['panoramic_rgb']: 64 x 252 x 3, observations['panoramic_depth']:  64 x 252 x 1, observations['target_goal']: 64 x 252 x 4
    # env_global_node: b x 1 x 512

    # features(xt): p(at|xt) = σ(FC(xt)) Size: num_processes x f_dim (512)
    
        features, rnn_hidden_states, preds, ffeatures, = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        x = distribution.logits

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action = action.squeeze(-1)
        
        if self.no_critic:
            return action, x, rnn_hidden_states
        
        action_log_probs = distribution.log_probs(action)
        value = self.critic(features) # uses a FC layer to map features to a scalar value of size num_processes x 1
        
        # The shape of the output should be B * N * (shapes)
        # NOTE: change distribution_entropy to x
        return value, action, action_log_probs, rnn_hidden_states, x, preds, ffeatures if return_features else None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        get the value of the current state which is represented by an observation
        """
        # features is the logits of action candidates
        features, *_ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, disable_forgetting=True
        )
        value = self.critic(features)
        return value

    def evaluate_actions(
            self, observations, rnn_hidden_states, env_global_node, prev_actions, masks, action
    ):
        features, rnn_hidden_states, preds, env_global_node, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, disable_forgetting=True
        )
        distribution = self.action_distribution(features)
        x = distribution.logits
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, preds[0], preds[1], rnn_hidden_states, x

    def get_forget_idxs(self):
        return self.net.perception_unit.forget_idxs

class CNNRNNNet(nn.Module):
    def __init__(
            self,
            config,
            observation_space,
            action_space,
            normalize_visual_inputs=False
    ):
        super().__init__()
        rnn_input_size = 0
        
        # Init the depth encoder
        assert config.CNNRNN.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder",
            "None",
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        if config.CNNRNN.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=config.CNNRNN.DEPTH_ENCODER.output_size,
                checkpoint=config.CNNRNN.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=config.CNNRNN.DEPTH_ENCODER.backbone,
                trainable=config.CNNRNN.DEPTH_ENCODER.trainable,
            )
            rnn_input_size += config.CNNRNN.DEPTH_ENCODER.output_size
        else:
            self.depth_encoder = None

        # Init the RGB visual encoder
        assert config.CNNRNN.RGB_ENCODER.cnn_type in [
            "ResnetRGBEncoder",
            "None",
        ], "RGB_ENCODER.cnn_type must be 'ResnetRGBEncoder'."

        if config.CNNRNN.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=config.CNNRNN.RGB_ENCODER.output_size,
                backbone=config.CNNRNN.RGB_ENCODER.backbone,
                trainable=config.CNNRNN.RGB_ENCODER.trainable,
                normalize_visual_inputs=normalize_visual_inputs,
            )
            rnn_input_size += 2 * config.CNNRNN.RGB_ENCODER.output_size
        else:
            self.rgb_encoder = None
            logger.info("RGB encoder is none")

        sem_seg_output_size = 0
        self.semantic_predictor = None
        self.is_thda = False
        self.use_semantic_encoder = config.CNNRNN.USE_SEMANTICS
        if config.CNNRNN.USE_SEMANTICS:
            sem_embedding_size = config.SEMANTIC_ENCODER.embedding_size

            self.is_thda = config.SEMANTIC_ENCODER.is_thda
            rgb_shape = observation_space.spaces["rgb"].shape
            spaces = {
                "semantic": Box(
                    low=0,
                    high=255,
                    shape=(rgb_shape[0], rgb_shape[1], sem_embedding_size),
                    dtype=np.uint8,
                ),
            }
            sem_obs_space = Dict(spaces)
            self.sem_seg_encoder = ResnetSemSeqEncoder(
                sem_obs_space,
                output_size=config.SEMANTIC_ENCODER.output_size,
                backbone=config.SEMANTIC_ENCODER.backbone,
                trainable=config.SEMANTIC_ENCODER.train_encoder,
                semantic_embedding_size=sem_embedding_size,
                is_thda=self.is_thda
            )
            sem_seg_output_size = config.SEMANTIC_ENCODER.output_size
            logger.info("Setting up Sem Seg model")
            rnn_input_size += sem_seg_output_size

            self.embed_sge = config.embed_sge
            if self.embed_sge:
                self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=self.obj_categories_embeddingdevice)
                self.mapping_mpcat40_to_goal = np.zeros(
                    max(
                        max(mapping_mpcat40_to_goal21.keys()) + 1,
                        50,
                    ),
                    dtype=np.int8,
                )

                for key, value in mapping_mpcat40_to_goal21.items():
                    self.mapping_mpcat40_to_goal[key] = value
                self.mapping_mpcat40_to_goal = torch.tensor(self.mapping_mpcat40_to_goal, device=self.device)
                rnn_input_size += 1

        self.use_gpscompss = config.USE_GPS_COMPASS
        if self.use_gpscompss:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")


        self.goal_embedding = nn.Sequential(
            nn.Linear(config.CNNRNN.RGB_ENCODER.output_size, config.CNNRNN.RGB_ENCODER.output_size),
            nn.ReLU(True)
        )

        logger.info("\n\nSetting up Object Goal sensor")

        self.use_prev_action = config.SEQ2SEQ.use_prev_action
        if config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        self.output_size = config.STATE_ENCODER.hidden_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=self.output_size,
            num_layers=config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=config.STATE_ENCODER.rnn_type,
        )
        
        self.train()

        s = 'Parameter number: {}\n'.format(sum(param.numel() for param in self.parameters()))
        s += '- RGB ResNet18 encoder: {}\n'.format(sum(param.numel() for param in self.rgb_encoder.parameters()))
        s += '- Depth ResNet50 encoder: {}\n'.format(sum(param.numel() for param in self.depth_encoder.parameters()))
        if hasattr(self, 'sem_seg_encoder'):
            s += '- Semantic ResNet18 encoder: {}\n'.format(sum(param.numel() for param in self.sem_seg_encoder.parameters()))
        s += '- RNN state encoder: {}\n'.format(sum(param.numel() for param in self.state_encoder.parameters()))

        print(s)

    # @property
    # def output_size(self):
    #     return self.config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].contiguous().flatten(start_dim=1)
            
            if len(observations["objectgoal"].size()) == 3:
                observations["objectgoal"] = observations["objectgoal"].contiguous().view(
                    -1, observations["objectgoal"].size(2)
                )

            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]
            if self.is_thda:
                idx = self.mapping_mpcat40_to_goal[idx].long()
            idx = idx.to(obj_semantic.device)

            if len(idx.size()) == 3:
                idx = idx.squeeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1)
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1)).float()
            return goal_visible_area.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        B = observations["rgb"].shape[0] # 同时包含观测和目标图像

        x = []

        if self.depth_encoder is not None:

            depth_embedding = self.depth_encoder(observations)
            x.append(depth_embedding)

        # Encode both obs and the target image
        if self.rgb_encoder is not None:
            observations["rgb"] = torch.cat([observations["rgb"], observations["target_goal"]], dim=0)
            rgb_target_embedding = self.rgb_encoder(observations)

            x.append(rgb_target_embedding.view(2, -1, rgb_target_embedding.shape[-1]).permute(1,0,2).contiguous().view(B, -1))

        if self.use_semantic_encoder != 0:
            semantic_obs = observations["semantic"]
            if len(semantic_obs.size()) == 4:
                observations["semantic"] = semantic_obs.contiguous().view(
                    -1, semantic_obs.size(2), semantic_obs.size(3)
                )
            if self.embed_sge:
                sge_embedding = self._extract_sge(observations)
                x.append(sge_embedding)

            sem_seg_embedding = self.sem_seg_encoder(observations)
            x.append(sem_seg_embedding)

        if self.use_gpscompss:
            obs_gps = observations['position']
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            
            x.append(self.gps_embedding(obs_gps))
        
            obs_compass = observations["rotation"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if self.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=1)
        
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states, None, None

