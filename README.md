# EmbodiedUNIT: Embodied Multi-task Learning for Visual Navigation
<!-- This repository is the official implementation of [MemoNav](https://arxiv.org/abs/2030.12345).  -->


<!-- ![Model overview](./assets/Main_Model.png) -->

## Requirements
The source code is developed and tested in the following setting. 
- Python 3.7
- pytorch 1.7.1
- habitat-sim 0.2.1
- habitat 0.2.1

Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation instructions.

To install requirements:

```
pip install -r requirements.txt
```

## Data Setup
### Scene Datasets
The scene datasets and task datasets used for training should be organized as follows:
```
Any path
  └── data
      └── scene_datasets
          └── gibson_habitat
          |   └── *.glb, *.navmeshs
          └── mp3d
              └── 1LXtFkjw3qL
              |    └── 1LXtFkjw3qL.glb
              |    └── 1LXtFkjw3qL.navmesh
              └── ... (other scenes)           
```

Then modify the task configuration file as follows so that the Habitat simulator can load these datasets:

```
For example, in objectnav_mp3d_il.yaml:
SCENES_DIR: "path/to/data/scene_datasets"
```

## Training and Evaluation Dataset Setup
```
Any path
  └── data
      └── datasets
      │   └── pointnav
      │   |   └── gibson
      │   |       └── v1
      │   |           └── train
      │   |           └── val
      |   └── objectnav
      |   │       └── mp3d
      |   │           └── mp3d_70k
      |   │                └── train
      |   │                |   └── train.json.gz
      |   │                |   └── content 
      |   │                |        └── 1LXtFkjw3qL.json.gz 
      |   │                └── val
      |   │                └── sample
      |   └── imagenav
      │   |   └── gibson
      │   |       └── v1
      │   |           └── train
      │   |           └── val
      |   └── VLN
      |   │   └── VLNCE_R2R
      |   │       └── sample
      |   │       └── train
      |   │       └── val
      |   └── imagenav
      └── scene_datasets
```

Then modify the task configuration file as follows so that the Habitat simulator can load these datasets:

```
For example, in objectnav_mp3d_il.yaml:
DATA_PATH: "path/to/data/datasets/objectnav/mp3d/mp3d_70k/{split}/{split}.json.gz"
```

### ImageNav
ImageNav datasets originate from [the Habitat-Web human demonstration dataset](https://github.com/Ram81/habitat-imitation-baselines#downloading-human-demonstrations-dataset), which is originally used for ObjectNav. We transform the target format from object category to image in this dataset to generate ImageNav training datasets.


Then modify the task configuration file as follows so that the Habitat simulator can load these datasets:
```
In ./configs/vistargetnav_mp3d.yaml:
DATA_PATH: "Your_habitat_lab_dir/data/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.json.gz"
```

### ObjectNav

We use [the Habitat-web 70k demonstrations](https://habitat-on-web.s3.amazonaws.com/release/datasets/objectnav/objectnav_mp3d_70k.zip) for training and [this official dataset](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip) for evaluation.


## Training

### Model Definition

(1) Baseline Model Architecture

The policy used in this project is the CNNRNN used in the Habitat-web paper and adapted for ImageNav and VLN.

You can find the pretrained RedNet semantic segmentation model weights [here](https://habitat-on-web.s3.amazonaws.com/release/checkpoints/rednet/rednet_semmap_mp3d_tuned.pth) and the pretrained depth encoder weights [here](https://habitat-on-web.s3.amazonaws.com/release/checkpoints/depth_encoder/gibson-2plus-resnet50.pth).

Please modify ```SEMANTIC_ENCODER.rednet_ckpt``` and ```DEPTH_ENCODER.ddppo_checkpoint``` in the config accordingly.

(2) Define your own model

Every navigation policy needs to be defined in ```custom_habitat_baselines/il/env_based/policy``` and must contains a class method named ```def act(self, *args)```. 

To specify the policy class you wiil use, please modify the entry called ```POLICY: ***``` in your model configuration file in the ```configs``` directory.




### Simulator Settings

The navigation environment is defined in ```custom_habitat_baselines/common/environments.py```.

The agent's sensors, measures, actions, tasks, and goals are defined in ```./custom_habitat/tasks/nav/nav.py```.

### Training Pipeline

The imitation learning pipieline is defined in ```./custom_habitat_baselines/il/env_based/il_trainer.py```

Use this command to train an agent for ImageNav:

```
python run.py --cfg ./configs/ImageNav/CNNRNN/CNNRNN_woPose_envbased.yaml --split train [--debug 1]
```

Use this command to train an agent for ObjectNav:

```
python run.py --cfg ./configs/ObjectNav/CNNRNN/Objnav_wopano.yaml --split train
```

<!-- ```train on multiple GPUs
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=[GPU_num] train_bc.py --config ./configs/CNNRNN/CNNRNN.yaml --stop
``` -->

## Evaluation

Use this command to evalluate an agent for ObjectNav:

```
python run.py --cfg ./configs/ObjectNav/CNNRNN/Objnav_wopano.yaml --run-type eval --split val --ckpt path/to/ckpt


## Pre-trained Models



## Results
TODO
<!-- Our model achieves the following performance on:

### [Gibson single-goal test dataset](https://github.com/facebookresearch/image-goal-nav-dataset)
Following the experiemntal settings in VGM, our MemoNav model was tested on 1007 samples of this dataset. We reported the performances of our model and various baselines in the table. (NOTE: we re-evaluated the VGM pretrained model and reported new results)

| Model name         | SR  | SPL |
| ------------------ |---------------- | -------------- |
| ANS   |     0.30         |      0.11       |
| Exp4nav   |     0.47         |      0.39       |
| SMT   |     0.56         |      0.40       |
| Neural Planner   |     0.42         |      0.27       |
| NTS   |     0.43         |      0.26       |
| VGM   |     0.75         |      0.58       |
| MemoNav (ours)   |     0.78         |      0.54       |

### [Gibson multi-goal test dataset](https://github.com/facebookresearch/image-goal-nav-dataset)
We compared our model with VGM on multi-goal test datasets which can be downloaded [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/hongxin_li_zju_edu_cn/EV8yJjE4PZRFjspQRUuK8SUBitWymCw7GCj-rMiWOCI18Q?e=FAGIHY).

| Model name         | 2goal PR  | 2goal PPL | 3goal PR  | 3goal PPL | 4goal PR  | 4goal PPL |
| ------------------ |---------------- | -------------- |---------------- | -------------- |---------------- | -------------- |
| VGM   |     0.45        |      0.18       | 0.33 | 0.08 | 0.28 | 0.05 |
| MemoNav (ours)   |     0.50         |      0.17       | 0.42 | 0.09 | 0.31 | 0.05 |

### Visualizations


https://user-images.githubusercontent.com/49870114/175005380-b3623e2b-22e5-4e1f-88e3-7dc41fe3ddec.mp4



https://user-images.githubusercontent.com/49870114/175005417-7939a6f2-987f-431d-b5b2-abac1141cdfb.mp4



https://user-images.githubusercontent.com/49870114/175005441-871eb72c-a938-4086-a699-d9dd4d8857f5.mp4




https://user-images.githubusercontent.com/49870114/175005452-cb3f720d-4143-4000-a673-b4172945fdb3.mp4 -->