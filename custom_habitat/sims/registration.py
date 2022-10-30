#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from custom_habitat.core.logging import logger
from custom_habitat.core.registry import registry
from custom_habitat.sims.habitat_simulator import _try_register_habitat_sim
from custom_habitat.sims.pyrobot import _try_register_pyrobot
from custom_habitat.sims.pickplace import _try_register_pickplace_sim


def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(
        id_sim
    )
    return _sim(**kwargs)


_try_register_habitat_sim()
_try_register_pyrobot()
_try_register_pickplace_sim()
