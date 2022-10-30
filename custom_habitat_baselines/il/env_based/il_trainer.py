#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import time

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from custom_habitat import Config, logger
from custom_habitat.utils import profiling_wrapper
from custom_habitat.utils.visualizations.utils import observations_to_image

from custom_habitat_baselines.common.base_trainer import BaseRLTrainer
from custom_habitat_baselines.common.environments import get_env_class
from custom_habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from custom_habitat_baselines.common.utils import poll_checkpoint_folder

from custom_habitat_baselines.il.env_based.common.rollout_storage import RolloutStorage
from custom_habitat_baselines.il.env_based.algos.agent import ILAgent
from custom_habitat_baselines.common.tensorboard_utils import TensorboardWriter
from custom_habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from custom_habitat_baselines.utils.env_utils import construct_envs
from custom_habitat_baselines.il.env_based.policy.rednet import load_rednet
from env_utils import *
from model.policy import CNNRNNPolicy

import cv2
import matplotlib.pyplot as plt

class ILEnvTrainer(BaseRLTrainer):
    r"""Trainer class for behavior cloning.
    """
    #supported_tasks = ["ObjectNav-v1"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_actor_critic_agent(self, config: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        model_config = self.config.MODEL
        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        self.policy = eval(config.POLICY)(
            config,
            observation_space,
            self.envs.action_spaces[0]
            )
        
        self.policy.to(self.device)

        optimizer = torch.optim.AdamW(
            [{'params': list(filter(lambda p: p.requires_grad, self.policy.parameters())),
            'initial_lr': config.BC.lr}],
            lr=config.BC.lr,
            eps=config.BC.eps,
        )
        
        ckpt_file = config.BC.CKPT
        if ckpt_file != '':
            ckpt = torch.load(config.BC.CKPT, map_location="cpu")
            self.policy.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

            self.ckpt_count = ckpt['extra_state'].get('count_checkpoints', -1)
            self.resume_steps = ckpt['extra_state']['step']
            print('############### Loading pretrained RL state dict (epoch {}) ###############'.format(self.resume_steps))
            
        self.semantic_predictor = None
        if model_config.USE_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=model_config.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=model_config.SEMANTIC_ENCODER.num_classes
            )
            self.semantic_predictor.eval()

        self.agent = ILAgent(
            model=self.policy,
            optimizer=optimizer,
            num_envs=self.envs.num_envs,
            num_mini_batch=config.RL.PPO.num_mini_batch,
            max_grad_norm=config.BC.max_grad_norm,
        )

    
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
        curr_checkpoint_list = [os.path.join(self.config.CHECKPOINT_FOLDER,x)
                                for x in os.listdir(self.config.CHECKPOINT_FOLDER)
                                if 'ckpt' in x]
        if len(curr_checkpoint_list) >= 45 :
            oldest_file = min(curr_checkpoint_list, key=os.path.getctime)
            os.remove(oldest_file)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "room_visitation_map", "exploration_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # fetch actions and environment state from replay buffer
        next_actions = rollouts.get_next_actions()
        actions = next_actions.long().unsqueeze(-1)
        step_data = [a.item() for a in next_actions.long().to(device="cpu")]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.envs.step(step_data)
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        if self.config.MODEL.USE_SEMANTICS and self.current_update >= self.config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            # Subtract 1 from class labels for THDA YCB categories
            if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                batch["semantic"] = batch["semantic"] - 1
        #batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward  # type: ignore
        running_episode_stats["count"] += 1 - masks  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v  # type: ignore

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            actions,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()

        total_loss, rnn_hidden_states = self.agent.update(rollouts)

        rollouts.after_update(rnn_hidden_states)

        return (
            time.time() - t_update_model,
            total_loss,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self, rank=0, debug=0) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        config = self.config
        profiling_wrapper.configure(
            capture_start_step=config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.envs = construct_envs(
            config, env_class=eval(config.ENV_NAME)
        )

        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID[rank])
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(config.CHECKPOINT_FOLDER):
            os.makedirs(config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(config)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        num_steps = 16 if debug else config.RL.PPO.num_steps

        rollouts = RolloutStorage(
            num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            config.MODEL.STATE_ENCODER.hidden_size,
            config.MODEL.STATE_ENCODER.num_recurrent_layers,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        #batch = apply_obs_transforms_batch(batch, self.obs_transforms)


        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
            # Use first semantic observations from RedNet predictor as well
            if sensor == "semantic" and config.MODEL.USE_SEMANTICS:
                semantic_obs = self.semantic_predictor(batch["rgb"], batch["depth"])
                # Subtract 1 from class labels for THDA YCB categories
                if config.MODEL.SEMANTIC_ENCODER.is_thda:
                    semantic_obs = semantic_obs - 1
                rollouts.observations[sensor][0].copy_(semantic_obs)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.RL.PPO.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        start_steps = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        count_checkpoints = 0 if not hasattr(self, 'ckpt_count') else self.ckpt_count + 1


        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, config.NUM_UPDATES),  # type: ignore
        )
        self.possible_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            num_updates = 3 if debug else config.NUM_UPDATES
            log_interval = 1 if debug else config.LOG_INTERVAL
            ckpt_interval = 1 if debug else config.CHECKPOINT_INTERVAL

            input(debug)
            input(num_updates)
            for update in range(num_updates):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                
                self.current_update = update

                if config.RL.PPO.use_linear_lr_decay and update > 0:
                    lr_scheduler.step()  # type: ignore

                if config.RL.PPO.use_linear_clip_decay and update > 0:
                    self.agent.clip_param = config.RL.PPO.clip_param * linear_decay(
                        update, config.NUM_UPDATES
                    )

                profiling_wrapper.range_push("rollouts loop")
                for _step in range(num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                # videos = rollouts.observations['rgb'][:,0].cpu().numpy().astype(np.uint8)
                # vout = cv2.VideoWriter('./test.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (videos[0].shape[1], videos[0].shape[0]))
                # for frame in videos:
                #     vout.write(frame)
                # vout.release()

                # break
                (
                    delta_pth_time,
                    total_loss
                ) = self._update_agent(config, rollouts)
                pth_time += delta_pth_time


                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [total_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["action"])},
                    count_steps,
                )

                # log stats
                if update % log_interval == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\tloss: {:.3f}".format(
                            update, (count_steps  - start_steps) / (time.time() - t_start), total_loss
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

   
                # if update == config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE - 1:
                #     self.save_checkpoint(
                #         f"ckpt_gt_best.{count_checkpoints}.pth",
                #         dict(step=count_steps),
                #     )

                # checkpoint model
                if update % ckpt_interval == 0:
                    self.save_checkpoint(
                        "ckpt{}_frame{}.pth".format(count_checkpoints, count_steps), dict(step=count_steps, count_checkpoints=count_checkpoints)
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()
    
    def eval(self, ckpt_dir) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID[0])
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(ckpt_dir):
                # evaluate singe checkpoint
                self._eval_checkpoint(ckpt_dir, writer)
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _make_results_dir(self, split="val"):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        dir_name = os.path.join(self.config.RESULTS_DIR, self.config.VERSION)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        self._make_results_dir(self.config.EVAL.SPLIT)

        # if self.config.EVAL.USE_CKPT_CONFIG:
        #     conf = ckpt_dict["config"]
        #     config = self._setup_eval_config(ckpt_dict["config"])
        # else:
        config = self.config.clone()

        # config = config.IL.BehaviorCloning

        # config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        # config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 500
        # config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(config, config.MODEL)

        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=True)    
        self.policy = self.agent.model
        self.policy.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        #batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.MODEL.STATE_ENCODER.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        current_episode_steps = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                if self.semantic_predictor is not None:
                    batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                    if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                        batch["semantic"] = batch["semantic"] - 1
                (
                    logits,
                    test_recurrent_hidden_states,
                ) = self.policy(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                )
                current_episode_steps += 1

                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            #batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_steps[i] = 0

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {"rgb": batch["rgb"][i]}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k not in ["reward", "pred_reward"]}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
