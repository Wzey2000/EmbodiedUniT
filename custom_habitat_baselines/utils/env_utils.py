#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Type, Union

#import habitat
from custom_habitat import Config, Env, RLEnv, VectorEnv, ThreadedVectorEnv, make_dataset, logger

from env_utils.make_env_utils import add_camera, add_panoramic_camera
def make_env_fn(
    config: Config, env_class: Union[Type[Env], Type[RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """

    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )

    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env


def construct_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES

    if isinstance(config.SIMULATOR_GPU_ID, list):
        gpus = config.SIMULATOR_GPU_ID
    else:
        gpus = [config.SIMULATOR_GPU_ID]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_processes

    configs = []
    env_classes = [env_class for _ in range(num_envs)]

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)#, config=config.TASK_CONFIG.DATASET)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_processes):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_processes) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED = proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            use_pano = "PANORAMIC_SENSOR" in task_config.TASK.SENSORS
            if use_pano:
                task_config = add_panoramic_camera(task_config)
            else:
                task_config = add_camera(task_config)
            print("\033[0;33;40m[env_utils.py] The agent uses panorama: {}\033[0m\n".format(use_pano))

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            # task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
            proc_config.freeze()
            configs.append(proc_config)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )

    return envs


def construct_ddp_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    workers_ignore_signals: bool = False,
    world_rank: int = 0,
    world_size: int = 0,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    gpu = config.SIMULATOR_GPU_ID[world_rank]

    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        # random.shuffle(scenes)
        num_scenes_per_node =  len(scenes) // world_size
        start_idx = world_rank * num_scenes_per_node
        end_idx = start_idx + num_scenes_per_node
        if world_rank == (world_size - 1):
            scenes = scenes[start_idx:]
        else:
            scenes = scenes[start_idx : end_idx]
        logger.info("world rank: {}, scenes: {} - {}".format(world_rank, len(scenes), scenes))


    scene_splits: List[List[str]] = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = [gpu]

        use_pano = "PANORAMIC_SENSOR" in task_config.TASK.SENSORS
        if use_pano:
            task_config = add_panoramic_camera(task_config)
        else:
            task_config = add_camera(task_config)
        print("\033[0;33;40m[env_utils.py] The agent uses panorama: {}\033[0m\n".format(use_pano))

        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
