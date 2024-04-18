FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    git python3-dev python3-pip \
    libglib2.0-0 

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

  # Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Dependency for EGL
RUN apt update && apt install -y cmake build-essential libgl1-mesa-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev

RUN pip install torch==2.2.0 \
torchvision \
torchaudio \
--index-url https://download.pytorch.org/whl/cu121

RUN pip install torch_geometric 

RUN pip install pyg_lib \
torch_scatter \
torch_sparse torch_cluster \
torch_spline_conv \
-f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN pip install mmcv==1.6.0 \
matplotlib \
argparse \
lpips \
plyfile \
imageio-ffmpeg \ 
h5py \
imageio \
natsort \ 
numpy \ 
wandb \
open3d \
seaborn \
trimesh  \
spatialmath-python \
pybind11 \
gym \
moviepy \
shapely