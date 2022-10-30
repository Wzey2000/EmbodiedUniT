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
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
    ReplayActionSpec, AgentStateSpec)


@registry.register_dataset(name="ObjectNav-v1")
class ObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    count = 0

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset
    
    @staticmethod
    def dedup_goals_dset(dataset):
        if len(dataset.episodes) == 0:
            return dataset
        goals_by_category = dict()
        for i, ep in enumerate(dataset.episodes):
            dataset.episodes[i].object_category = ep.goals[0].object_category
            # ep = ObjectGoalNavEpisode(**ep)
            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals
            dataset.episodes[i].goals = []
        dataset.goals_by_category = goals_by_category
        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = "{}_{}".format(episode.scene_id.split("/")[-1].split(".")[0], str(i))

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]
        # random.shuffle(self.episodes)


@registry.register_dataset(name="ObjectNav-v2")
class ObjectNavDatasetV2(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {'couch': 'sofa', 'toilet': 'toilet', 'bed': 'bed', 'tv': 'tv_monitor', 'potted plant': 'plant', 'chair': 'chair'}

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        self.max_replay_steps = config.MAX_REPLAY_STEPS
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            if "_shortest_path_cache" in episode:
                del episode["_shortest_path_cache"]
            
            if "gibson" in episode["scene_id"]:
                episode["scene_id"] = "gibson_semantic/{}".format(episode["scene_id"].split("/")[-1])

            episode = ObjectGoalNavEpisode(**episode)
            # episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if not episode.is_thda:
                episode.goals = self.goals_by_category[episode.goals_key]
                if episode.scene_dataset == "gibson":
                    episode.object_category = self.gibson_to_mp3d_category_map[episode.object_category]
            else:
                goals = []
                for g in episode.goals:
                    g = ObjectGoal(**g)
                    for vidx, view in enumerate(g.view_points):
                        view_location = ObjectViewLocation(**view)  # type: ignore
                        view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
                        g.view_points[vidx] = view_location
                    goals.append(g)
                episode.goals = goals

                objects = [ObjectInScene(**o) for o in episode.scene_state["objects"]]
                scene_state = [SceneState(objects=objects).__dict__]
                episode.scene_state = scene_state

            if episode.reference_replay is not None:
                for i, replay_step in enumerate(episode.reference_replay):
                    # replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
                    replay_step["agent_state"] = None
                    episode.reference_replay[i] = ReplayActionSpec(**replay_step)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)
            
            if len(episode.reference_replay) > self.max_replay_steps:
                continue

            self.episodes.append(episode)  # type: ignore [attr-defined]

