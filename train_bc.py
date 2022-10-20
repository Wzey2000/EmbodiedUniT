from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from configs.default import get_config
from model.policy import *
from trainer.il.bc_trainer import BC_trainer
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import os
import argparse
from dataset.multidemodataset import HabitatDemoMultiGoalDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

torch.backends.cudnn.enable = True
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from custom_habitat_baselines.utils.common import linear_decay

os.environ['HABITAT_SIM_LOG'] = "quiet"

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 train_bc.py --cfg 
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml containing info about experiment")
parser.add_argument("--stop", action='store_true', default=False, help="include stop action or not",)
parser.add_argument('--data-dir', default='/data/hongxin_li/Habitat_web/ImageNav', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--video', default=0, type=int)
# distributed training
parser.add_argument('--gpus',
                    help='gpus id for multiprocessing training',
                    type=str)
parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://127.0.0.1:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--rank',
                    default=0,
                    type=int,
                    help='node rank for distributed training')
# For DDP
parser.add_argument('--local_rank', default=0, type=int,
                help='node rank for distributed training')
args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#device = 'cpu' if args.gpu == '-1' else 'cuda'

def train():
    version_name = args.config.split('/')[-1][:-len(".yaml")] + "_IL"

    config = get_config(args.config, version_name)

    observation_space = SpaceDict({
        'rgb': Box(low=0, high=256, shape=(240, 320, 3), dtype=np.float32),
        'depth': Box(low=0, high=256, shape=(240, 320, 1), dtype=np.float32),
        'target_goal': Box(low=0, high=256, shape=(240, 320, 3), dtype=np.float32),
        'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
        'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
        'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32),
        "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        "gps": Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(config.TASK_CONFIG.TASK.GPS_SENSOR.DIMENSIONALITY,),
            dtype=np.float32),
    })
    #device = torch.device('cpu' if config.TORCH_GPU_ID == '-1' else 'cuda:' + str(config.TORCH_GPU_ID))

    s = time.time()

    action_space = Discrete(6) if args.stop else Discrete(5)
    stop_info = 'INCLUDING' if args.stop else 'EXCLUDING'
    print('POLICY : {}'.format(config.POLICY))
    print('TRAINING INFO : {} STOP ACTION'.format(stop_info))

    config.defrost()
    
    if args.debug != 0:
        config.BC.batch_size = 8

    config.NUM_PROCESSES = config.BC.batch_size
    
    #config.TORCH_GPU_ID = args.gpu
    config.freeze()

    policy = eval(config.POLICY)(
        observation_space=observation_space,
        action_space=action_space,
        no_critic=True,
        normalize_visual_inputs=True,
        config=config
    )
    policy.train()
    optim = torch.optim.Adam(
            [{'params': list(filter(lambda p: p.requires_grad, policy.parameters())),
            'initial_lr': config.BC.lr}],
            lr=config.BC.lr,
            eps=config.BC.eps
        )
    
    DATA_DIR = args.data_dir
    train_data_list = [os.path.join(DATA_DIR, 'train', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'train')))]
    valid_data_list = [os.path.join(DATA_DIR, 'val', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'val')))]

    params = {'batch_size': config.BC.batch_size,
              'shuffle': True,
              'num_workers': config.BC.num_workers,
              'pin_memory': True}

    train_dataset = HabitatDemoMultiGoalDataset(config, train_data_list, args.stop)
    train_dataloader = DataLoader(train_dataset, **params)
    num_batches = len(train_dataloader)
    train_iter = iter(train_dataloader)

    valid_dataset = HabitatDemoMultiGoalDataset(config, valid_data_list, args.stop)
    valid_params = params

    valid_dataloader = DataLoader(valid_dataset, **valid_params)
    valid_iter = iter(valid_dataloader)

    start_step = 0
    start_epoch = 1 # starting from 1 instead of 0
    if args.resume:
        checkpoint = torch.load(args.resume)#, map_location=torch.device('cpu'))
        start_epoch, start_step = checkpoint['trained']
        policy.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        print('load {}, ckpt_ep {}, ckpt_step {}'.format(args.resume, start_epoch, start_step))

        if start_step % num_batches != 0:
            start_step = start_step // num_batches * num_batches # retrain from this epoch
        else:
            start_epoch += 1 # train from next epoch

        # if 'train_global_steps' in checkpoint.keys() and \
        # 'valid_global_steps' in checkpoint.keys():
        #     writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        #     writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']


    # Distributed Computing
    gpus = config.TORCH_GPU_ID
    master = True
    if config.DISTRIBUTED: # This block is not available
        # args.local_rank+=int(gpus[0])

        device = gpus[args.local_rank]
        print('This process is using GPU', device)
        master = device == int(gpus[0])
        dist.init_process_group(backend='nccl')
        # if config.MODEL.SYNC_BN:
        #     policy = nn.SyncBatchNorm.convert_sync_batchnorm(policy)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpus is not None:
            torch.cuda.set_device(device)
            policy.cuda(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # workers = int(workers / ngpus_per_node)
            policy = nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=True
            )
        else:
            policy.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            policy = nn.parallel.DistributedDataParallel(policy)
    else: # implement this block
        gpu_ids = eval('['+gpus+']')
        device = gpu_ids[0]
        print('This process is using GPU', str(device))
        policy = nn.DataParallel(policy, gpu_ids).cuda(device)
    
    trainer = eval(config.IL_TRAINER_NAME)(config, policy, optim, device)

    # version_name = config.saving.name if args.version == 'none' else args.version
    # version_name = version_name
    # version_name += '_start_time:{}'.format(time.ctime())

    if args.video:
        IMAGE_DIR = os.path.join('data', 'images', version_name)
        # SAVE_DIR = os.path.join('data', 'new_checkpoints', version_name)
        # LOG_DIR = os.path.join('data', 'logs', version_name)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        # os.makedirs(SAVE_DIR, exist_ok=True)
        # os.makedirs(LOG_DIR, exist_ok=True)

    with_env_global_node = config.GCN.WITH_ENV_GLOBAL_NODE
    respawn_env_global_node = config.GCN.RESPAWN_GLOBAL_NODE
    randominit_env_global_node = config.GCN.RANDOMINIT_ENV_GLOBAL_NODE
    global_node_featdim = config.features.visual_feature_dim

    
    # create or load an env global node
    env_global_node = None
    # if with_env_global_node:
    #     if args.resume != '':
    #         env_global_node = checkpoint.get(
    #         'env_global_node', None
    #     )
    #     if env_global_node is None:
    #         env_global_node = torch.randn(1, global_node_featdim) if randominit_env_global_node else torch.zeros(1, global_node_featdim)
    
    print_every = config.saving.log_interval if args.debug == 0 else 1
    save_every = config.saving.save_interval if args.debug == 0 else 2
    eval_every = config.saving.eval_interval if args.debug == 0 else 2

    if master:
        writer = SummaryWriter(log_dir=config.TENSORBOARD_DIR)

    start = time.time()
    temp = start

    step = start_step
    step_values = [10000, 50000, 100000]
    step_index = 0
    lr = config.BC.lr

    def adjust_learning_rate(optimizer, step_index, lr_decay):
        # lr = config.BC.lr * (lr_decay ** step_index)
        lr = config.BC.lr * (lr_decay ** step_index)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    #trainer.to(device)
    #trainer.train()

    max_epoch = config.BC.max_epoch if args.debug == 0 else 3

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optim,
            lr_lambda=lambda x: linear_decay(x, max_epoch * num_batches),
            last_epoch = start_step if args.resume else -1
        )

    if master:
        print(30*"="+"\n [Training Params] Max epoch: {}, Batch size: {}, {} batches per epoch\n".format(max_epoch, config.BC.batch_size, num_batches) + 30*"=")
        print('Num params: {}'.format(sum(param.numel() for param in policy.parameters())))

    for epoch in range(start_epoch, max_epoch+1):
        train_dataloader = DataLoader(train_dataset, **params)
        train_iter = iter(train_dataloader)
        loss_summary_dict = {}
        
        for batch in train_iter:
            if with_env_global_node:
                env_global_node = env_global_node.unsqueeze(0).repeat(batch[0].shape[0], 1, 1).to(device) # NUM_PROCESSES x 1 x 512

            results, loss_dict, new_env_global_node = trainer.train(batch, env_global_node)

            lr = optim.state_dict()['param_groups'][0]['lr']

            # if step in step_values:
            # step_index += 1
            # lr = adjust_learning_rate(trainer.optim, step_index, config.BC.lr_decay)

            if master:
                for k,v in loss_dict.items():
                    if k not in loss_summary_dict.keys():
                        loss_summary_dict[k] = []
                    loss_summary_dict[k].append(v)
                
                if step % print_every == 0:
                    loss_str = ''
                    writer_dict = {}
                    for k,v in loss_summary_dict.items():
                        value = np.array(v).mean()
                        loss_str += '%s: %.3f, '%(k,value)
                        writer_dict[k] = value
                    print("[Training] epoch: %d/%d, step %d/%d, time: %.2fm, , lr: %.3e, %ds per step || loss:" % (epoch, max_epoch, step % num_batches + 1, num_batches, (time.time() - start) // 60,
                                                                                                    lr, (time.time() - temp) / print_every), loss_str)
                    loss_summary_dict = {}
                    temp = time.time()
                    writer.add_scalars('loss', writer_dict, step)
                    if args.video:
                        trainer.visualize(results, os.path.join(IMAGE_DIR, 'train_{}_{}'.format(results['scene'],step)))

                if step % save_every == 0:
                    trainer.save(
                        file_name=os.path.join(config.CHECKPOINT_FOLDER, 'epoch%04diter%05d.pt' % (epoch, step)),
                        env_global_node=new_env_global_node.mean(0) if new_env_global_node is not None else None, epoch=epoch, step=step)

                if step % eval_every == 0 and step > 0:
                    trainer.agent.eval()
                    eval_start = time.time()
                    with torch.no_grad():
                        val_loss_summary_dict = {}

                        for j in range(100 if args.debug == 0 else 5):
                            try:
                                batch = next(valid_iter)
                            except:
                                valid_dataloader = DataLoader(valid_dataset, **valid_params)
                                valid_iter = iter(valid_dataloader)
                                batch = next(valid_iter)

                            eval_env_global_node = None
                            if with_env_global_node:
                                if respawn_env_global_node:
                                    eval_env_global_node = torch.randn(1, 1, global_node_featdim) if randominit_env_global_node else torch.zeros(1, 1, global_node_featdim)
                                else:
                                    eval_env_global_node = new_env_global_node.mean(0, keepdim=True)
                                eval_env_global_node = eval_env_global_node.repeat(batch[0].shape[0], 1, 1).to(device)
                            
                            results, loss_dict, _ = trainer.train(batch, eval_env_global_node, train=False)

                            if j % 100 == 0 and args.video:
                                trainer.visualize(results,os.path.join(IMAGE_DIR, 'validate_{}_{}_{}'.format(results['scene'], step, j)))
                            for k, v in loss_dict.items():
                                if k not in val_loss_summary_dict.keys():
                                    val_loss_summary_dict[k] = []
                                val_loss_summary_dict[k].append(v)

                        loss_str = ''
                        writer_dict = {}
                        for k, v in val_loss_summary_dict.items():
                            value = np.array(v).mean()
                            loss_str += '%s: %.3f ' %(k, value)
                            writer_dict[k] = value
                        print("[validation] epoch: %d/%d, step %d/%d time: %.2fm, lr: %.3e, %ds per step || loss: " % (
                            epoch, max_epoch, step % num_batches + 1, num_batches, (time.time() - start) // 60, lr, 
                            (time.time() - eval_start) / print_every), loss_str)
                        loss_summary_dict = {}
                        temp = time.time()
                        writer.add_scalars('val_loss', writer_dict, step)

                    trainer.agent.train()
            
            lr_scheduler.step()
            step += 1
            
            if with_env_global_node:
                if respawn_env_global_node: # reset env global node to 
                    env_global_node = torch.randn(1, global_node_featdim) if randominit_env_global_node else torch.zeros(1, global_node_featdim)
                else:
                    env_global_node = new_env_global_node.detach().mean(0)

            if args.debug != 0 and step % eval_every == 0: break

    if master:
        writer.close()                
    print('===> end training. Time elapsed: {:.2f}m'.format((time.time() - start) // 60))

if __name__ == '__main__':
    train()
