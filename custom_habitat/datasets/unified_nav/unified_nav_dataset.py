#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from custom_habitat.config import Config
from custom_habitat.core.dataset import Dataset, ALL_SCENES_MASK
from custom_habitat.core.registry import registry
from custom_habitat.datasets.utils import VocabDict
from custom_habitat.tasks.nav.nav import NavigationGoal, InstructionData, VLNEpisode, ObjectGoalNavEpisode, UnifiedNavEpisode

from custom_habitat.core.dataset import ObjectInScene, SceneState
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

from custom_habitat.tasks.nav.nav import (
    ImageGoal,
    NavigationEpisode,
    NavigationGoal,
    ReplayActionSpec,
    AgentStateSpec
)

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="UnifiedNav-v1")
class UnifiedDatasetV1(PointNavDatasetV1):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List
    instruction_vocab: VocabDict
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {'couch': 'sofa', 'toilet': 'toilet', 'bed': 'bed', 'tv': 'tv_monitor', 'potted plant': 'plant', 'chair': 'chair'}

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return (os.path.exists(
            config.OBJECTNAV_DATA_PATH.format(split=config.SPLIT)
        ) or os.path.exists(
            config.VLN_DATA_PATH.format(split=config.SPLIT)
        )) and os.path.exists(config.SCENES_DIR)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.goals_by_category = {}
        self.max_replay_steps = 500

        if config is None:
            return

        self.max_replay_steps = config.MAX_REPLAY_STEPS

        datasetfile_path_lst = [config.OBJECTNAV_DATA_PATH, config.VLN_DATA_PATH]

        # self.datasetfile_path = datasetfile_path
        for datasetfile_path in datasetfile_path_lst:
            if datasetfile_path == "": continue

            datasetfile_path = datasetfile_path.format(split=config.SPLIT)
            print('\033[0;36;40m[unified_nav_dataset] Loading dataset from: {}\033[0m\n'.format(datasetfile_path))

            with gzip.open(datasetfile_path, "rt") as f:
                if 'objectnav' in datasetfile_path.lower():
                    self.episodes.extend(self.from_ObjectNav_json(f.read(), scenes_dir=config.SCENES_DIR))
                if 'vln' in datasetfile_path.lower():
                    self.episodes.extend(self.from_VLN_json(f.read(), scenes_dir=config.SCENES_DIR))
            
            dataset_dir = os.path.dirname(datasetfile_path)
            has_individual_scene_files = os.path.exists(
                self.content_scenes_path.split("{scene}")[0].format(
                    data_path=dataset_dir
                )
            )
            if has_individual_scene_files:
                scenes = config.CONTENT_SCENES # may be [*] which means all scenes
                if ALL_SCENES_MASK in scenes:
                    scenes = self._get_scenes_from_folder(
                        content_scenes_path=self.content_scenes_path,
                        dataset_dir=dataset_dir,
                    )

                for scene in scenes:
                    scene_filename = self.content_scenes_path.format(
                        data_path=dataset_dir, scene=scene
                    )
                    with gzip.open(scene_filename, "rt") as f:
                        # from_json in this class is rewritten in child classes (e.g. ImageNavDatasetV1)
                        if 'objectnav' in datasetfile_path.lower():
                            self.episodes.extend(self.from_ObjectNav_json(f.read(), scenes_dir=config.SCENES_DIR))
                        if 'vln' in datasetfile_path.lower():
                            self.episodes.extend(self.from_VLN_json(f.read(), scenes_dir=config.SCENES_DIR))

            else:
                self.episodes = list(
                    filter(self.build_content_scenes_filter(config), self.episodes)
                )
            

        # if config.DATASET.VLN_DATA_PATH != "":
        #     dataset_filename = config.VLN_DATA_PATH.format(split=config.SPLIT)
        #     with gzip.open(dataset_filename, "rt") as f:
        #         self.episodes.extend(self.from_VLN_json(f.read(), scenes_dir=config.SCENES_DIR))
        # sself.episodes = list(self.episodes)
        # self.episodes = list(
        #     filter(self.build_content_scenes_filter(config), self.episodes)
        # )

    def from_ObjectNav_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ):
        episodes = []
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
            return episodes

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            
            if episode.get('reference_replay', None) is not None and len(episode['reference_replay']) > self.max_replay_steps:
                continue
            
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
            
            

            episodes.append(episode)  # type: ignore [attr-defined]
        return episodes
    
    def from_VLN_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        episodes = []
        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            if episode.get('reference_replay', None) is not None and len(episode['reference_replay']) > self.max_replay_steps:
                continue

            episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            
            if episode.reference_replay is not None:
                for i, replay_step in enumerate(episode.reference_replay):
                    # replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
                    replay_step["agent_state"] = None
                    episode.reference_replay[i] = ReplayActionSpec(**replay_step)
                    scene_state = [SceneState().__dict__] # Dummy SceneState
                    episode.scene_state = scene_state

            episodes.append(episode)
        
        return episodes

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
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g
