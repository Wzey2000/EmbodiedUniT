import argparse
import json, gzip
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--ep-per-env', type=int, default=200, help='number of episodes per environments')
parser.add_argument('--num-procs', type=int, default=4, help='number of processes to run simultaneously')
parser.add_argument("--gpu", type=str, default="0", help="gpus",)
parser.add_argument('--split', type=str, default="train", choices=['train','val'], help='data split to use')
parser.add_argument('--traj-dir', type=str, default='/data/hongxin_li/Habitat_web/datasets/objectnav/objectnav_mp3d_70k/train/content')
parser.add_argument('--data-dir', type=str, default="/data/hongxin_li/Habitat_web", help='directory to save the collected data')
args = parser.parse_args()
import os
os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import glob
import numpy as np
import habitat, habitat_sim
import habitat.sims
import habitat.sims.habitat_simulator
from habitat.utils.visualizations import maps, utils

import joblib
from configs.default import get_config
from env_utils.task_search_env import MultiSearchEnv
from tqdm import tqdm
import cv2
import matplotlib
TRAJ_DIR = args.traj_dir
CONTENT_PATH = os.path.join(habitat.__path__[0],'../data/datasets/pointnav/gibson/v1/train/content/')
NUM_EPISODE_PER_SPACE = args.ep_per_env
MAX_TRAJ_LEN = 200
TASK = 'ObjectNav'

task_splits = {
    'ImageNav': [0, 0.01],
    'ObjectNav': [0.25, 0.26],
    'PointNav': [0.5, 0.6],
    'REVERIE': [0.75, 0.85]
}
dataset_split = 0.9
recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
cmap = matplotlib.cm.get_cmap("rainbow")
act_dict = {
    "stop": 0,
    "move_forward": 1,
    "turn_left": 2,
    "turn_right": 3,
    "look_up": 4,
    "look_down": 5     
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "stop": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),     
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        v = (point[0] - bounds[0][0]) / meters_per_pixel
        u = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([int(u), int(v)]))
    return points_topdown

def make_env_fn(config_env, rank):
    config_env.defrost()
    config_env.SEED = rank * 1121
    config_env.freeze()
    env = MultiSearchEnv(config=config_env)
    env.seed(rank * 1121)
    return env

def data_collect(sim, agent, traj_list, config, scene_name):
    total = len(traj_list)
    left_idx, right_idx = int(total * task_splits[TASK][0]), int(total * task_splits[TASK][1])
    right_idx = max(right_idx, left_idx + 16)
    split_idx = int(dataset_split * (right_idx - left_idx))
    if args.split == 'train':
        right_idx = left_idx + split_idx
    elif 'val' in args.split:
        left_idx = left_idx + split_idx
    
    cnt = 1
    with tqdm(total=right_idx-left_idx) as pbar:
        for i in range(left_idx, right_idx):
            save_path = os.path.join(scene_name + '_{}.dat.gz'.format(i))
            
            if not os.path.exists(save_path):
                episode = traj_list[i]
                # the first element in reference_replay is the agent's initial pose
                # The last element is useless so the slice length is plut 1 
                agent_poses = episode["reference_replay"][-MAX_TRAJ_LEN-1:]

                idx = 0
                traj_dict = {'position': [], 'rotation': [], 'action': [],
                    'rgb': [], 'depth': [],
                    'target_idx': [0] * MAX_TRAJ_LEN, 'target': [], 'target_pose': [], 'target_rot': [],
                    'distance': []}

                # meters_per_pixel = maps.calculate_meters_per_pixel(1024 // 4, sim, sim.pathfinder) 
                # img = maps.get_topdown_map(sim.pathfinder, height=agent_poses[0]["agent_state"]["position"][1], meters_per_pixel=meters_per_pixel) # 256 x 505 x 3 uint8 numpy array
                # img = recolor_map[img]
                # 因为sim.step(action)会穿模，所以这里不在采用仿真器中智能体的位姿，而是直接从Habitat-web数据集中获取
                while idx < len(agent_poses) - 1:
                    agent_state = agent.get_state()
                    agent_state.position = np.array(agent_poses[idx]["agent_state"]["position"])  # world space
                    agent_state.rotation = np.array(agent_poses[idx]["agent_state"]["rotation"])  # world space
                    agent.set_state(agent_state)

                    action = agent_poses[idx]["action"].lower()
                    observations = sim.step('stop')
                    # agent.set_state(agent_state)

                    rgb = observations["color_sensor"][:,:,:-1] # 480 x 640 x 4 (uint8)
                    depth = observations["depth_sensor"][:,:,np.newaxis] # 480 x 640 x 1

                    if idx < len(agent_poses) - 1:
                        traj_dict['rgb'].append(rgb)
                        traj_dict['depth'].append(depth)
                        traj_dict['position'].append(np.stack([-agent_state.position[2], agent_state.position[0]])) # Only retain x,z coords
                        traj_dict['rotation'].append(agent_state.rotation)

                        path = habitat_sim.ShortestPath()
                        path.requested_start = agent_state.position
                        path.requested_end = agent_poses[-1]["agent_state"]["position"]
                        sim.pathfinder.find_path(path) # bool
                        geodesic_dist = path.geodesic_distance
                        traj_dict['distance'].append(geodesic_dist)
                    if idx >=1:
                        traj_dict['action'].append(act_dict[action])

                    #agent_state = agent.get_state()
                    # print(idx, '/', len(agent_poses) - 1, ', Goal:{}'.format(episode['object_category']), ', Action:', action, ', Pos:', agent_state.position)

                    # if idx >=1:
                    #     traj_points = convert_points_to_topdown(sim.pathfinder, traj_dict['position'], meters_per_pixel)
                    #     color = tuple(map(lambda x: int(255 * x), cmap(idx /len(agent_poses) - 1)))
                    #     print(len(traj_points), idx, traj_points[-1])
                    #     cv2.line(img,
                    #             (traj_points[idx-1][1], traj_points[idx-1][0]),
                    #             (traj_points[idx][1], traj_points[idx][0]),
                    #             color=color,
                    #             thickness = 2)
                    
                    # img_t = cv2.resize(img, None, fx=480/256, fy=480/256)

                    #v.append(np.concatenate([rgb, img_t], axis=1))
                    idx += 1
                
                # vout = cv2.VideoWriter('./test.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (v[0].shape[1], v[0].shape[0]))
                # for frame in v:
                #     vout.write(frame)
                # vout.release()
                if TASK == 'ImageNav':
                    traj_dict['target'].append(copy.deepcopy(traj_dict['rgb'][-1]))
                elif TASK == 'ObjectNav':
                    traj_dict['target'].append(copy.deepcopy(episode['object_category']))

                traj_dict['target_pose'].append(copy.deepcopy(traj_dict['position'][-1]))
                traj_dict['target_rot'].append(copy.deepcopy(traj_dict['rotation'][-1]))
                traj_dict['action'].append(act_dict['stop'])
                joblib.dump(traj_dict, save_path)

            pbar.update(1)
            pbar.set_description('[%s] %03d/%03d data collected' % (scene_name.split('\\')[-1],cnt,right_idx-left_idx))

            cnt += 1
def main():
    split = args.split
    DATA_DIR = args.data_dir + '/' +TASK
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    DATA_DIR = os.path.join(DATA_DIR, split)
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

    config = get_config(config_paths=r'./configs/vgm_mp3d_test.yaml')
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.defrost()

    """
    For the episode to be considered successful, the agent must stop within 1m Euclidean distance of the goal object within a maximum of 500 steps and be able to turn to view the object from that end position
    """
    config.RL.SUCCESS_DISTANCE = 1.0
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 300
    config.TASK_CONFIG.ENVIRONMENT.NUM_GOALS = 1
    #config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["GOAL_INDEX"] + config.TASK_CONFIG.TASK.MEASUREMENTS
    config.TASK_CONFIG.TASK.GOAL_INDEX = config.TASK_CONFIG.TASK.SPL.clone()
    config.TASK_CONFIG.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
    config.DIFFICULTY = 'random'
    config.noisy_actuation = False
    config.freeze()
    #scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    space_files = os.listdir(TRAJ_DIR)
    
    print([i[:-len('.json.gz')] for i in space_files])

    scene_dir = '/data/jing_li/habitat-lab/data/scene_datasets/mp3d'
    
    print(20*'-' + '\nCollecting trajectories in {} scenes\nSaving in {}\n'.format(scene_dir, DATA_DIR))
    for space_id, space_file in enumerate(space_files):
        scene_name = space_file.split('./')[-1][:-len('.json.gz')]
        sim_settings = {
        "width": 320,  # Spatial resolution of the observations
        "height": 240,
        "scene": os.path.join(scene_dir, scene_name, '{}.glb'.format(scene_name)),  # Scene path
        "default_agent": 0,
        "sensor_height": 0.88,  # Height of sensors in meters
        "color_sensor": True,  # RGB sensor
        "depth_sensor": True,  # Depth sensor
        "semantic_sensor": False,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
        }
        
        cfg = make_cfg(sim_settings)
        sim = habitat_sim.Simulator(cfg)
        sim.seed(sim_settings["seed"])
        sim.pathfinder.seed(1)
        agent = sim.initialize_agent(sim_settings["default_agent"])

        space_name = space_file[:-len('.json.gz')]
        try:
            print('=' * 50)
            print('[%03d/%03d] %s' % (space_id, len(space_files), space_name))
            config.defrost()
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = [space_name]
            config.freeze()

            with gzip.open(os.path.join(TRAJ_DIR, space_file), 'rb') as g:
                traj_list = json.load(g) # dict_keys(['goals_by_category', 'episodes', 'category_to_task_category_id', 'category_to_mp3d_category_id'])
            
            data_collect(sim, agent, traj_list['episodes'], config, os.path.join(DATA_DIR, space_name))
        except:
            raise
            print('{} failed may be the space is too large or unexpected error'.format(space))
            unexpected_skip.append(space)
            print('unexpected_skipped envs : ', unexpected_skip)
        
        sim.close()
if __name__ == "__main__":
    main()
