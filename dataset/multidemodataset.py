import torch.utils.data as data
import numpy as np
import joblib
import torch
import quaternion as q
import time
import pickle

class HabitatDemoMultiGoalDataset(data.Dataset):
    def __init__(self, cfg, data_list, include_stop = False):
        self.data_list = data_list
        self.img_size = (240, 320)
        self.action_dim = 6 if include_stop else 5
        self.max_demo_length = cfg.BC.max_demo_length
        self.use_aux_tasks = cfg.USE_AUXILIARY_INFO
        self.calc_pose = cfg.USE_GPS_COMPASS or cfg.USE_AUXILIARY_INFO
        self.use_inflection_weighting = cfg.BC.USE_IW

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        with open(self.data_list[index], 'rb') as f:
            demo_data = pickle.load(f)
        #demo_data = joblib.load(self.data_list[index])
        # print(['{}: {}'.format(k, len(v)) for k,v in demo_data.items()])
        #scene = self.data_list[index].split('/')[-1].split('_')[0]
        # start_pose = [demo_data['position'][0], q.as_float_array(demo_data['rotation'][0])]
        rotation = q.as_euler_angles(demo_data['rotation'])[:,1]
        target_indices = demo_data['target_idx']

        orig_data_len = len(demo_data['position'])
        start_idx = np.random.randint(orig_data_len - 10) if orig_data_len > 10 else 0
        end_idx = - 1

        demo_rgb = demo_data['rgb'][start_idx:end_idx] # float32
        demo_length = np.minimum(len(demo_rgb), self.max_demo_length)

        demo_dep = demo_data['depth'][start_idx:end_idx]# float32

        demo_rgb_out = np.zeros([self.max_demo_length, *demo_rgb.shape[1:]])
        demo_rgb_out[:demo_length] = demo_rgb[:demo_length]
        demo_dep_out = np.zeros([self.max_demo_length, *demo_dep.shape[1:]])
        demo_dep_out[:demo_length] = demo_dep[:demo_length]

        demo_act = demo_data['action'][start_idx:start_idx+demo_length]
        demo_act_out = np.ones([self.max_demo_length]) * (-100)

        demo_act_out[:demo_length] = demo_act -1 if self.action_dim == 5 else demo_act

        targets = np.zeros([self.max_demo_length]) # (Max_len,) int 64
        targets[:demo_length] = target_indices[start_idx:start_idx+demo_length]
        target_img= demo_data['target_img']
        target_img_out = np.zeros([5, *target_img[0].shape])
        target_img_out[:len(target_img)] = target_img

        return_tensor = {
            'rgb':torch.from_numpy(demo_rgb_out).float(), # float
            'depth':torch.from_numpy(demo_dep_out).float(), # float
            'action':torch.from_numpy(demo_act_out).float(), # float
            'target_idx':targets,
            'target_img':torch.from_numpy(target_img_out).float() # float
        }

        if self.calc_pose:
            positions = np.zeros([self.max_demo_length,2])
            positions[:demo_length] = demo_data['position'][start_idx:start_idx+demo_length]
            rotations = np.zeros([self.max_demo_length])
            positions[:demo_length] = demo_data['position'][start_idx:start_idx+demo_length]
            rotations[:demo_length] = rotation[start_idx:start_idx+demo_length]

            return_tensor['position'] = torch.from_numpy(positions)
            return_tensor['rotation'] = torch.from_numpy(rotations)
        
        if self.use_aux_tasks:
            have_been = np.zeros([self.max_demo_length])
            for idx, pos_t in enumerate(positions[start_idx:end_idx]):
                if idx == 0:
                    have_been[idx] = 0
                else:
                    dists = np.linalg.norm(positions[start_idx:end_idx][:idx - 1] - pos_t,axis=1)
                    if len(dists) > 10:
                        far = np.where(dists > 1.0)[0]
                        near = np.where(dists[:-10] < 1.0)[0]
                        if len(far) > 0 and len(near) > 0 and (near < far.max()).any():
                            have_been[idx] = 1
                        else:
                            have_been[idx] = 0
                    else:
                        have_been[idx] = 0
            
            return_tensor['distance'] = np.zeros([self.max_demo_length])
            distances = np.maximum(1-demo_data['distance'][start_idx:start_idx+demo_length]/2.,0.0)
            return_tensor['distance'][:demo_length] = torch.from_numpy(distances).float()
            return_tensor['have_been'] = torch.from_numpy(have_been).float()
        
        if self.use_inflection_weighting:
            iw_coefs = np.ones(demo_act_out.shape[0])
            iw_idxs = np.concatenate([np.ones(shape=(1,), dtype=bool), demo_act_out[:-1] != demo_act_out[1:]], axis=0)

            iw = demo_act_out.shape[0] / (1 + iw_idxs.sum())
            iw_coefs[iw_idxs] = iw
            iw_coefs /= iw_coefs.sum()

            return_tensor['IW'] = iw_coefs
        
        return return_tensor

