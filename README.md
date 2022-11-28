# nerf2nerf: Pairwise Registration of Neural Radiance Fields
### [Project Page](https://nerf2nerf.github.io/) | [Video](https://youtu.be/S071rGezdNM) | [Paper](https://arxiv.org/abs/2211.01600)

<img src="https://github.com/nerf2nerf/nerf2nerf.github.io/raw/main/video/iterations.gif" height=200>

PyTorch implementation of nerf2nerf, a framework for robustly registering two NeRFs with respect to a common object of interest.

[nerf2nerf: Pairwise Registration of Neural Radiance Fields](https://nerf2nerf.github.io/)  
 [Lily Goli](https://lilygoli.github.io/),
 [Daniel Rebain](http://drebain.com/),
 [Sara Sabour](https://ca.linkedin.com/in/sara-sabour-63019132),
 [Animesh Garg](https://animesh.garg.tech/),
 [Andrea Tagliasacchi](https://taiya.github.io/)

## Quick Start
To get started quickly, assuming [Singularity](https://docs.sylabs.io/guides/2.6/user-guide/installation.html#installation) and NVIDIA drivers/libraries are installed on your device, get the bash script file from [here](https://drive.google.com/file/d/1gYrSP4k03LXPtfwVIC9joOK6kzUIHbJO/view?usp=sharing) and put it in target directory. Then run the command below in terminal to get the demo running. 
Please note the script will clone this repository and a Singlarity image for a compatible environment, automatically and there is no need for separate cloning. Tested to work with nvidia-cuda-toolkit 11.0.
```sh
./run_demo.sh
```
You can view resulting images and graphs in your browser on localhost:6006 and track sampling process in 3D on localhost:5946.

## Build Environment and Run

#### Dataset
You can download the dataset from [here](https://drive.google.com/drive/folders/1jNpwAv1T1ntjIHUMJ1wABePA2Z8_nRRQ?usp=sharing) and put the 'scenes' directoy right under the main directory.
#### Environment

Set up a conda environment and activate:

```sh
conda create -n n2n python=3.10
conda activate n2n
pip install -r requirements.txt
pip install git+https://github.com/jonbarron/robust_loss_pytorch
```
#### Run
To run registration code:
```sh
python main.py --yaml <object_name> 
# example: python main.py --yaml bust 
```
The results are saved to tensorboard. Connect to tensorboard to view results.
To reobtain the distilled models and run the surface field distillation code:

```sh
python distill.py --scene_no <scene_number> --a_or_b <a/b>
# example: python distill.py --scene_no 1 --a_or_b a
```

## Visdom
If use_vis is enabled in options yaml file, to see the sample points changing during registration in 3D, in a terminal run:
```sh
conda activate n2n
visdom -p <port_number>
# example: visdom -p 5946
```
This will launch visdom on localhost:<port_number>. You can change the port_number in option yaml files.
## Tensorboard
To see the results, launch tensorboard in terminal:
```sh
tensorboard --logdir="." --port=<tensorboard_port_number> --bind_all
```
Then open a browser window (or alternatively use port forwarding if you are using a remote server) and view results on localhost:<tensorboard_port_number>.
