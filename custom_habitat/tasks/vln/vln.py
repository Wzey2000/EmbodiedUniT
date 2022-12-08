#!/usr/bin/env python3

# Owners/maintainers of the Vision and Language Navigation task:
#   @jacobkrantz: Jacob Krantz
#   @koshyanand: Anand Koshy

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import attr
from gym import spaces

from custom_habitat.core.registry import registry
from custom_habitat.core.simulator import Observations, Sensor
from custom_habitat.core.utils import not_none_validator
from custom_habitat.tasks.nav.nav import NavigationEpisode, NavigationTask



@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    r"""Vision and Language Navigation Task
    Goal: An agent must navigate to a goal location in a 3D environment
        specified by a natural language instruction.
    Metric: Success weighted by Path Length (SPL)
    Usage example:
        examples/vln_reference_path_follower_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
