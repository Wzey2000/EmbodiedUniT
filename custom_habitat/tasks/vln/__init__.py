#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from custom_habitat.core.embodied_task import EmbodiedTask
from custom_habitat.core.registry import registry


def _try_register_vln_task():
    try:
        from custom_habitat.tasks.vln.vln import VLNTask  # noqa: F401
    except ImportError as e:
        vlntask_import_error = e

        @registry.register_task(name="VLN-v0")
        class VLNTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise vlntask_import_error
