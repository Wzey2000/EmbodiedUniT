# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
from cv2 import log
import numpy as np
from gym import spaces

from custom_habitat.config import Config
from custom_habitat.core.dataset import SceneState
from custom_habitat.core.logging import logger
from custom_habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from custom_habitat.core.utils import not_none_validator
from custom_habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask
)

try:
    from custom_habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass


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
