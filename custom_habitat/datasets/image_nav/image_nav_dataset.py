#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence

from custom_habitat.config import Config
from custom_habitat.core.dataset import ObjectInScene, SceneState
from custom_habitat.core.registry import registry
from custom_habitat.core.simulator import AgentState, ShortestPathPoint
from custom_habitat.core.utils import DatasetFloatJSONEncoder
from custom_habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from custom_habitat.tasks.nav.nav import (
    ImageGoal,
    NavigationEpisode,
    NavigationGoal,
    ReplayActionSpec,
    AgentStateSpec
)


@registry.register_dataset(name="VisTargetNav-v1")
class VisTargetNavDatasetV1(PointNavDatasetV1):
    def __init__(self, config: Optional[Config] = None, filter_fn= None) -> None:
        self.filter_fn = filter_fn
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)
            if self.filter_fn is not None and not self.filter_fn(episode): continue
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                # NOTE: modification
                processed_goal = {"position": goal["position"], "radius": goal["radius"]}
                
                episode.goals[g_index] = NavigationGoal(**processed_goal)
            #if episode.shortest_paths is not None:
            #    for path in episode.shortest_paths:
            #        for p_index, point in enumerate(path):
            #            path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)


@registry.register_dataset(name="ImageNav-v1")
class ImageNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    episodes: List[NavigationEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def __init__(self, config: Optional[Config] = None) -> None:
        print('\033[0;36;40m[image_nav_dataset] Initializing ImageNav-v1...\033[0m\n')
        if config is not None:
            self.max_replay_steps = config.MAX_REPLAY_STEPS
        else:
            self.max_replay_steps = 500
        super().__init__(config)
        self.episodes = list(self.episodes)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        print('[ImageNav-v1] Loading {} episodes\n'.format(len(deserialized["episodes"])))
        for episode in deserialized["episodes"]:
            replay = episode['reference_replay']
            if len(replay) > self.max_replay_steps:
                continue

            for k in ['object_category', 'is_thda', 'scene_state', 'scene_dataset']:
                if k in episode.keys():
                    del episode[k]

            episode = NavigationEpisode(**episode)
            # episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # Only handle single-goal tasks
            episode.goals = [
                ImageGoal(
                    position = replay[-1]["agent_state"]["position"],
                    rotation = replay[-1]["agent_state"]["rotation"])
            ]

            for i, replay_step in enumerate(replay):
                # replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
                replay_step["agent_state"] = AgentStateSpec(
                    position=replay_step["agent_state"]["position"],
                    rotation=replay_step["agent_state"]["rotation"]
                    )
                replay[i] = ReplayActionSpec(**replay_step)

            # if episode.shortest_paths is not None:
            #     for path in episode.shortest_paths:
            #         for p_index, point in enumerate(path):
            #             if point is None or isinstance(point, (int, str)):
            #                 point = {
            #                     "action": point,
            #                     "rotation": None,
            #                     "position": None,
            #                 }

            #             path[p_index] = ShortestPathPoint(**point)
            
            self.episodes.append(episode)  # type: ignore [attr-defined]

