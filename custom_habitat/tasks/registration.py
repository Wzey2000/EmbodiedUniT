#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from custom_habitat.core.logging import logger
from custom_habitat.core.registry import registry
# from custom_habitat.tasks.eqa import _try_register_eqa_task
from custom_habitat.tasks.nav import _try_register_nav_task
from custom_habitat.tasks.vln import _try_register_vln_task
# from custom_habitat.tasks.pickplace.pickplace import PickPlaceTask


def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(
        id_task
    )

    return _task(**kwargs)

# from custom_habitat.tasks.nav.object_nav_task import ObjectNavDatasetV1, ObjectNavDatasetV2
# from custom_habitat.tasks.nav.nav import NavigationTask
#_try_register_eqa_task()
_try_register_nav_task()
# _try_register_vln_task()
# _try_register_pickplace_task()
