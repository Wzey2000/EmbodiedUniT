#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from custom_habitat.core.logging import logger
from custom_habitat.core.registry import registry
#from .image_nav import *
#from .object_nav import *
# from custom_habitat.datasets.eqa import _try_register_mp3d_eqa_dataset
#from custom_habitat.datasets.object_nav import _try_register_objectnavdatasetv1
# from custom_habitat.datasets.pointnav import _try_register_pointnavdatasetv1
# from custom_habitat.datasets.unified_nav import _try_register_unified_nav_dataset
# from custom_habitat.datasets.pickplace import _try_register_pickplace_dataset



def make_dataset(id_dataset, **kwargs):
    
    _dataset = registry.get_dataset(id_dataset)

    logger.info("Initializing dataset {} (class name: {})".format(id_dataset, _dataset))
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)  # type: ignore

from custom_habitat.datasets.object_nav.object_nav_dataset import (  # noqa: F401
            ObjectNavDatasetV1,
        )


# _try_register_objectnavdatasetv1()
# _try_register_mp3d_eqa_dataset()
# _try_register_pointnavdatasetv1()
# _try_register_r2r_vln_dataset()
# _try_register_pickplace_dataset()
