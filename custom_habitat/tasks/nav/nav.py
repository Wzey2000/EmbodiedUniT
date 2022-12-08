#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here
import os, gzip, json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)
import attr
import numpy as np
from numpy import ndarray
import torch
import quaternion as q
from gym import spaces
from gym.spaces.box import Box

from dtw import dtw
from fastdtw import fastdtw

import habitat_sim
from custom_habitat.config import Config
from custom_habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from custom_habitat.core.logging import logger
from custom_habitat.core.registry import registry
from custom_habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorTypes,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from custom_habitat.core.utils import not_none_validator, try_cv2_import
from custom_habitat.sims.habitat_simulator.actions import HabitatSimActions
from custom_habitat.tasks.utils import cartesian_to_polar, get_habitat_sim_action, merge_sim_episode_config
from custom_habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from custom_habitat.utils.visualizations import fog_of_war, maps

try:
    from custom_habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

from custom_habitat.core.dataset import Dataset, Episode, SceneState


cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 128

task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}



@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)

@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)

"""
Goal Definitions
"""
@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None

@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None

@attr.s(auto_attribs=True, kw_only=True)
class ImageGoal(NavigationGoal):
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = 1.0

@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float] = 0.

@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None

@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None

"""
Episode Definitions
"""

@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_replay: Optional[List[ReplayActionSpec]] = None
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None

@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state: Optional[List[SceneState]] = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "mp3d"

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        reference_path: List of (x, y, z) positions which gives the reference
            path to the goal that aligns with the instruction.
        instruction: single natural language instruction guide to goal.
        trajectory_id: id of ground truth trajectory path.
    """
    reference_path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_replay: Optional[List[ReplayActionSpec]] = None
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)

@attr.s(auto_attribs=True, kw_only=True)
class UnifiedNavEpisode(Episode):
    subtask = ""
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None
    scene_state: Optional[List[SceneState]] = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "mp3d"
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"

"""
Sensor Definitions
"""

class HabitatSimSensor:
    sim_sensor_type: habitat_sim.SensorType
    _get_default_spec = Callable[..., habitat_sim.sensor.SensorSpec]
    _config_ignore_keys = {"height", "type", "width"}

def check_sim_obs(obs: Optional[ndarray], sensor: Sensor) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )

@registry.register_sensor#(name="CustomRGBSensor")
class HabitatSimRGBSensor(RGBSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    RGBSENSOR_DIMENSION = 3

    def __init__(self, config: Config, **kwargs: Any) -> None:
        #self.sim = sim
        #self.agent_id = config.AGENT_ID
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.config = config
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.HEIGHT,
                self.config.WIDTH,
                self.RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, observations: Dict[str, Union[ndarray, bool, "Tensor"]],*args: Any, **kwargs: Any
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], observations.get(self.uuid, None))
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]
        return obs

@registry.register_sensor#(name="CustomDepthSensor")
class HabitatSimDepthSensor(DepthSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    _config_ignore_keys = {
        "max_depth",
        "min_depth",
        "normalize_depth",
    }

    min_depth_value: float
    max_depth_value: float

    def __init__(self, config: Config, **kwargs: Any) -> None:
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
        
        #self.agent_id = config.AGENT_ID
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(
        self, observations: Dict[str, Union[ndarray, bool, "Tensor"]],*args: Any, **kwargs: Any
    ) -> VisualObservation:
        
        obs = observations.get(self.uuid, None)
        check_sim_obs(obs, self)

        # 突然多了一维
        # if isinstance(obs, np.ndarray):
        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)

        if len(obs.shape) == 2:
            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        # else:
        #     obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)  # type: ignore[attr-defined]

        #     obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        return obs

@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor):
    sim_sensor_type: habitat_sim.SensorType
    sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.config = config
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.uint32,
        )

    def get_observation(
        self, observations: Dict[str, Union[ndarray, bool, "Tensor"]],*args: Any, **kwargs: Any
    ) -> VisualObservation:
        obs = observations.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs

@registry.register_sensor(name="PanoramicPartRGBSensor")
class PanoramicPartRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartSemanticSensor")
class PanoramicPartSemanticSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartDepthSensor")
class PanoramicPartDepthSensor(DepthSensor):
    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.angle = config.ANGLE
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + self.angle

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = np.expand_dims(
                obs, axis=2
            )
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )
        return obs

@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim = sim
        self.agent_id = config.AGENT_ID
        super().__init__(config=config)
        self.config = config
        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        if isinstance(observations['rgb_0'][:,:,:3], torch.Tensor):
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = torch.cat(rgb_list, 1)
        else:
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = np.concatenate(rgb_list, 1)
        if rgb_array.shape[1] > self.config.HEIGHT*4:
            left = rgb_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            rgb_array = rgb_array[:,slice:slice+self.config.HEIGHT*4]
        return rgb_array
        #return make_panoramic(observations['rgb_left'],observations['rgb'],observations['rgb_right'], self.torch)

@registry.register_sensor(name="PanoramicDepthSensor")
class PanoramicDepthSensor(DepthSensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.agent_id = config.AGENT_ID
        if config.NORMALIZE_DEPTH: self.depth_range = [0,1]
        else: self.depth_range = [config.MIN_DEPTH, config.MAX_DEPTH]
        self.min_depth_value = config.MIN_DEPTH
        self.max_depth_value = config.MAX_DEPTH
        self.num_camera = config.NUM_CAMERA
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.depth_range[0],
            high=self.depth_range[1],
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['depth_%d' % (i)] for i in range(self.num_camera)]
        if isinstance(observations['rgb_0'][:,:,:3], torch.Tensor):
            depth_array = torch.cat(depth_list, 1)
            depth_array = torch.clamp(depth_array, min=self.min_depth_value, max=self.max_depth_value)
            #depth_array = depth_array.unsqueeze(2)
        else:
            depth_array = np.concatenate(depth_list, 1)
            depth_array = np.clip(depth_array, self.min_depth_value, self.max_depth_value)
            #depth_array = np.expand_dims(depth_array, axis=2)

        if depth_array.shape[1] > self.config.HEIGHT*4:
            left = depth_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            depth_array = depth_array[:,slice:slice+self.config.HEIGHT*4]

        #if self.config.NORMALIZE_DEPTH:
        #   depth_array = (depth_array - self.min_depth_value)/(self.max_depth_value - self.min_depth_value)
        return depth_array

@registry.register_sensor(name="PanoramicSemanticSensor")
class PanoramicSemanticSensor(SemanticSensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.agent_id = config.AGENT_ID
        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic"
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['semantic_%d'%(i)] for i in range(self.num_camera)]
        return np.concatenate(depth_list,1)

HabitatSimVizSensors = Union[
    HabitatSimRGBSensor, HabitatSimDepthSensor, HabitatSimSemanticSensor
]

# Used for panoramic setting
@registry.register_sensor(name="PanoImageGoalSensor")
class PanoImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "image_goal"

    def __init__(
        self, sim, config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goals = []
        for i in range(len(episode.goals)):
            goal_position = np.array(episode.goals[0].position, dtype=np.float32)
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2 ** 32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            goal_observation = self._sim.get_observations_at(
                position=goal_position.tolist(), rotation=source_rotation
            )
            goals.append(goal_observation[self._rgb_sensor_uuid])
        return np.stack(goals)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id
        
        return self._current_image_goal

# used for 79-degree FOV setting
@registry.register_sensor(name="ImageGoalSensor")
class ImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "image_goal"

    def __init__(
        self, sim, config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goals = []
        for i in range(len(episode.goals)):
            goal_position = episode.goals[0].position
            goal_rotation = episode.goals[0].rotation

            goal_observation = self._sim.get_observations_at(
                position=goal_position, rotation=goal_rotation
            )
            goals.append(goal_observation[self._rgb_sensor_uuid])
        
        return np.stack(goals)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id
        
        return self._current_image_goal

@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.GOAL_SPEC_MAX_VAL - 1
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )
            logger.info("max object cat: {}".format(max_value))
            logger.info("cats: {}".format(self._dataset.category_to_task_category_id.values()))

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_sensor(name="PanoramicVisTargetSensor")
class PanoramicVisTargetSensor(Sensor):
    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        use_depth = True#'DEPTH_SENSOR_0' in self._sim.config.sim_cfg.AGENT_0.SENSORS
        use_rgb = True#'RGB_SENSOR_0' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = config.HEIGHT
        self.width = 252
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        self.num_camera = config.NUM_CAMERA
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "image_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1.0, shape=(self.height, self.width, self.channel), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        episode_id = episode.episode_id
        scene_id = episode.scene_id
        if (self.curr_episode_id != episode_id) or (self.curr_scene_id != scene_id):
            self.curr_episode_id = episode_id
            self.curr_scene_id = scene_id
            self.goal_obs = []
            self.goal_pose = []
            for goal in episode.goals:
                position = goal.position
                euler = [0, 2 * np.pi * np.random.rand(), 0]
                rotation = q.from_rotation_vector(euler)
                obs = self._sim.get_observations_at(position,rotation)
                rgb_list = [obs['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
                if isinstance(obs['rgb_0'], torch.Tensor):
                    rgb_array = torch.cat(rgb_list, 1) / 255.
                else:
                    rgb_array = np.concatenate(rgb_list, 1)/255.

                if rgb_array.shape[1] > self.height*4:
                    left = rgb_array.shape[1] - self.height*4
                    slice = left // 2
                    rgb_array = rgb_array[:, slice:slice + self.height*4]
                depth_list = [obs['depth_%d' % (i)] for i in range(self.num_camera)]
                if isinstance(obs['depth_0'], torch.Tensor):
                    depth_array = torch.cat(depth_list, 1)
                else:
                    depth_array = np.concatenate(depth_list, 1)
                if depth_array.shape[1] > self.height*4:
                    left = depth_array.shape[1] - self.height*4
                    slice = left // 2
                    depth_array = depth_array[:, slice:slice + self.height*4]
                if 'semantic_0' in obs.keys():
                    semantic_list = [obs['semantic_%d' % (i)] for i in range(self.num_camera)]
                    semantic_array = np.expand_dims(np.concatenate(semantic_list, 1),2)
                if isinstance(obs['depth_0'], torch.Tensor):
                    goal_obs = torch.cat([rgb_array, depth_array],2)
                else:
                    goal_obs = np.concatenate([rgb_array, depth_array],2)
                if 'semantic_0' in obs.keys():
                    goal_obs = np.concatenate([goal_obs, semantic_array],2)
                self.goal_obs.append(goal_obs)
                self.goal_pose.append([position, euler])
            if len(episode.goals) >= 1:
                if isinstance(obs['rgb_0'], torch.Tensor):
                    self.goal_obs = torch.stack(self.goal_obs,0)
                else:
                    self.goal_obs = np.array(self.goal_obs)
        return self.goal_obs


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    cls_uuid: str = "demonstration"
    def __init__(self, **kwargs):
        self.uuid = "demonstration"
        self.observation_space = spaces.Discrete(1)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        # Fetch next action as observation
        if task.is_resetting:  # reset
            self.timestep = 1
        
        if episode.reference_replay is not None and self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="InflectionWeightSensor")
class InflectionWeightSensor(Sensor):
    cls_uuid = "inflection_weight"
    def __init__(self, config: Config, **kwargs):
        self.uuid = "inflection_weight"
        self.observation_space = spaces.Discrete(1)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        if task.is_resetting:  # reset
            self.timestep = 0
        
        inflection_weight = 1.0
        if self.timestep == 0 or episode.reference_replay is None:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            inflection_weight = 1.0 
        elif episode.reference_replay[self.timestep - 1].action != episode.reference_replay[self.timestep].action:
            inflection_weight = self._config.INFLECTION_COEF
        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )

@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "heading"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())

@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        #self.observation_space = spaces.Discrete(0)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.int64).min,
            high=np.finfo(np.int64).max,
            shape=(200,),
            dtype=np.int64,
        )

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

"""
Measures
"""

@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        success_distance = self._config.SUCCESS_DISTANCE

        # if hasattr(episode, "is_thda") and episode.is_thda:
        #     success_distance = self._config.THDA_SUCCESS_DISTANCE

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure(name='Goal_Index')
class GoalIndex(Measure):
    cls_uuid: str = "goal_index"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.num_goals = len(episode.goals)
        self.goal_index = 0
        self.all_done = False
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self._metric = {'curr_goal_index': self.goal_index,
                        'num_goals': self.num_goals}

    def increase_goal_index(self):
        # when the agent has finished navigating to the final goal, self.goal_index will exceed the max index of goals。
        # Such implementation aims at correctly calculating Success_MultiGoal
        self.goal_index += 1
        self._metric = {'curr_goal_index': self.goal_index,
                        'num_goals': self.num_goals}
        self.all_done = self.goal_index >= self.num_goals
        return self.all_done


@registry.register_measure(name='Success_woSTOP')
class Success_woSTOP(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """
    cls_uuid: str = "success"

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
           distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0

@registry.register_measure(name='Success_MultiGoal')
class Success_MultiGoal(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """
    cls_uuid: str = "success"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, GoalIndex.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):     
        cur_goal_info = task.measurements.measures[
            GoalIndex.cls_uuid
        ].get_metric()

        self._metric = cur_goal_info['curr_goal_index'] / cur_goal_info['num_goals']

@registry.register_measure(name='Custom_DistanceToGoal')
class CustomDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        self.goal_idx = -1

        # For ObjectNav
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]

        self.update_metric(episode=episode, *args, **kwargs)


    def update_metric(self, episode: Episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        if GoalIndex.cls_uuid in task.measurements.measures:
            #input(task.measurements.measures[GoalIndex.cls_uuid].get_metric())
            self.goal_idx = task.measurements.measures[GoalIndex.cls_uuid].get_metric()['curr_goal_index']
        else:
            self.goal_idx = 0

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    episode.goals[self.goal_idx].position,
                    episode,
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                if hasattr(episode, "is_thda") and episode.is_thda:
                    distance_to_target = np.inf
                    for goal in episode.goals:
                        distance_to_goal = self._euclidean_distance(
                            current_position, goal.position
                        )
                        distance_to_target = min(distance_to_target, distance_to_goal)
                else:
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, self._episode_view_points, episode
                    )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            
            if distance_to_target == np.inf:
                logger.info('\033[0;33;40m inf  \033[0m')
            if not self._sim.pathfinder.is_navigable(episode.goals[self.goal_idx].position):
                logger.info('\033[0;33;40m unnav  \033[0m')

            self._metric = distance_to_target if distance_to_target != np.inf and self._sim.pathfinder.is_navigable(episode.goals[self.goal_idx].position) else 999.0


@registry.register_measure(name='DistanceToGoal')
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                    episode,
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target

@registry.register_measure(name='Custom_SPL')
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        #rint(self._start_end_episode_distance)
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric =(
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_measure(name='Custom_PPL')
class PPL(Measure):
    r"""PPL (Success weighted by Path Length)

    ref: MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation - Wani et. al
    
    The measure depends on Goal_index, Distance to Goal measure and Success measure
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._previous_goal_idx = -1
        self._start_end_episode_distance = []
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        # episode is a NavigationEpisode instance (defined in habitat.tasks.nav.nav.NavigationEpisode)
        # its keys are extracted from the *.json files in test set

        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, GoalIndex.cls_uuid, Success_MultiGoal.cls_uuid, ]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = episode['geodesic_distance'] # a list containing the geodesic distances between every two goals

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        progress = task.measurements.measures[
            Success_MultiGoal.cls_uuid
        ].get_metric()
        
        cur_goal_idx = task.measurements.measures[GoalIndex.cls_uuid].get_metric()['curr_goal_index']
        
        shortest_pathlen = sum(self._start_end_episode_distance[:cur_goal_idx+1])
        self._metric =(
            progress * shortest_pathlen
            / max(
                shortest_pathlen, self._agent_episode_distance
            )
        )

@registry.register_measure(name='Custom_SoftSPL')
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )



@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position
        
        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

        # print(self._start_end_episode_distance, self._agent_episode_distance, self._metric)

@registry.register_measure
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""
    cls_uuid = "top_down_map"

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count: Optional[int] = None
        self._map_resolution = config.MAP_RESOLUTION
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.DRAW_BORDER,
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            agent_position = self._sim.get_agent_state().position
            for goal in episode.goals:
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            # if abs(agent_position[1] - view_point.agent_state.position[1]) > 0.3:
                            #     continue
                            self._draw_point(
                                view_point.agent_state.position,
                                maps.MAP_VIEW_POINT_INDICATOR,
                            )
                except AttributeError:
                    pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:
            for goal in episode.goals:
                agent_position = self._sim.get_agent_state().position
                # if abs(agent_position[1] - goal.position[1]) > 0.3:
                #     continue
                try:
                    if len(episode.goals) == 2 and goal.info["is_receptacle"] == True:
                        self._draw_point(
                            goal.position, maps.MAP_RECEPTACLE_COLOR
                        )
                    else:
                        self._draw_point(
                            goal.position, maps.MAP_TARGET_POINT_INDICATOR
                        )
                except AttributeError:
                    pass

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            _shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)

        self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )

def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL)
    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self._config.SUCCESS_DISTANCE)


@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)

@registry.register_measure
class StepsTaken(Measure):
    """Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0

@registry.register_measure
class NDTW(Measure):
    """NDTW (Normalized Dynamic Time Warping)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "ndtw"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.dtw_func = fastdtw if config.FDTW else dtw


        with gzip.open(
            config.GT_PATH.format(split=config.SPLIT), "rt"
        ) as f:
            self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance
            / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
    """SDTW (Success Weighted be nDTW)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
        )
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = ep_success * nDTW

"""
Action
"""

@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.TURN_LEFT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.TURN_LEFT)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.TURN_RIGHT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.TURN_RIGHT)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "STOP"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_UP,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_DOWN,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: List[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()  # type: ignore

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False

    def overwrite_sim_config(self, sim_config, episode):
        super().overwrite_sim_config(sim_config, episode)

        sim_config.defrost()
        sim_config.scene_state = episode.scene_state
        sim_config.freeze()
        
        return sim_config

    def _check_episode_is_active(self,  action, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
