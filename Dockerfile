#####################################################
# gym-fish Env Dockerfile 
# NVIDIA Container Toolkit required 
# Refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide 
# 
# Build Container Image 
# > cd $DIR_OF_THE_DOCKERFILE 
# > docker build -t gym-fish . 
# 
# Create A New Container and Forwarding port 8888 for Jupyter 
# > docker run -p 8888:8888 -it --gpus all --name docker-name gym-fish 
# > // use the token to set a password for the jupyter notebook at the first time 
# 
# Start Docker & Stop Docker 
# > docker start docker-name 
# > // docker start -i docker-name (show interactive window attached to STDIN) 
# > docker stop docker-name 
# 
# You can access jupyter notebook at localhost:8888 on your browser 
#####################################################

# Setup Ubuntu 18.04 with Cuda 10.2
FROM nvidia/cuda:10.2-runtime-ubuntu18.04

# Set Environment Variable for CUDA
ENV PATH=${PATH}:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# Set Virtual Display
ENV DISPLAY=:99

# Helper Scripts
## Ubuntu Mirror scripts
RUN mkdir /root/util
RUN echo "#!/bin/sh" > /root/util/set-mirror-ubuntu.sh \
    && echo "touch /etc/apt/sources.list && cp /etc/apt/sources.list /etc/apt/sources.list.bak" >> /root/util/set-mirror-ubuntu.sh \
    && echo "sed -i 's#archive.ubuntu.com#mirrors.aliyun.com#g' /etc/apt/sources.list" >> /root/util/set-mirror-ubuntu.sh \
    && echo "sed -i 's#security.ubuntu.com#mirrors.aliyun.com#g' /etc/apt/sources.list" >> /root/util/set-mirror-ubuntu.sh
RUN echo "#!/bin/sh" > /root/util/unset-mirror-ubuntu.sh \
    && echo "rm /etc/apt/sources.list" >> /root/util/unset-mirror-ubuntu.sh \
    && echo "touch /etc/apt/sources.list.bak && mv /etc/apt/sources.list.bak /etc/apt/sources.list" >> /root/util/unset-mirror-ubuntu.sh

## Pypi Mirror scripts
RUN mkdir /root/.pip
RUN echo "#!/bin/sh" > /root/util/set-mirror-pypi.sh \
    && echo "touch /root/.pip/pip.conf && cp /root/.pip/pip.conf /root/.pip/pip.conf.bak" >> /root/util/set-mirror-pypi.sh \
    && echo "touch /root/.pip/pip.conf" >> /root/util/set-mirror-pypi.sh \
    && echo "echo '[global]' >> /root/.pip/pip.conf" >> /root/util/set-mirror-pypi.sh \
    && echo "echo 'index-url = https://mirrors.aliyun.com/pypi/simple/' >> /root/.pip/pip.conf" >> /root/util/set-mirror-pypi.sh \
    && echo "echo 'trusted-host = mirrors.aliyun.com' >> /root/.pip/pip.conf" >> /root/util/set-mirror-pypi.sh
RUN echo "#!/bin/sh" > /root/util/unset-mirror-pypi.sh \
    && echo "touch /root/.pip/pip.conf.bak && mv /root/.pip/pip.conf.bak /root/.pip/pip.conf" >> /root/util/unset-mirror-pypi.sh

## Conda Mirror scripts
RUN echo "#!/bin/sh" > /root/util/set-mirror-conda.sh \ 
    && echo "touch /root/.condarc && cp /root/.condarc /root/.condarc.bak" >> /root/util/set-mirror-conda.sh \
    && echo "echo 'channels:' > /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  - defaults' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo 'show_channel_urls: true' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo 'default_channels:' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo 'custom_channels:' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh \
    && echo "echo '  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /root/.condarc" >> /root/util/set-mirror-conda.sh
RUN echo "#!/bin/sh" > /root/util/unset-mirror-conda.sh \
    && echo  "touch /root/.condarc.bak && mv /root/.condarc.bak /root/.condarc" >> /root/util/unset-mirror-conda.sh

# Mirror scripts
RUN echo "#!/bin/sh" > /root/util/set-mirrors.sh \
    && echo ". /root/util/set-mirror-ubuntu.sh" >> /root/util/set-mirrors.sh \
    && echo ". /root/util/set-mirror-pypi.sh" >> /root/util/set-mirrors.sh \
    && echo ". /root/util/set-mirror-conda.sh" >> /root/util/set-mirrors.sh
RUN echo "#!/bin/sh" > /root/util/unset-mirrors.sh \
    && echo ". /root/util/unset-mirror-ubuntu.sh" >> /root/util/unset-mirrors.sh \
    && echo ". /root/util/unset-mirror-pypi.sh" >> /root/util/unset-mirrors.sh \
    && echo ". /root/util/unset-mirror-conda.sh" >> /root/util/unset-mirrors.sh

# Update gym-fish scripts
RUN echo "#!/bin/sh" > /root/util/update-gym-fish.sh \
    && echo "cd /root/gym-fish" >> /root/util/update-gym-fish.sh \
    && echo "git pull" >> /root/util/update-gym-fish.sh
RUN echo "#!/bin/sh" > /root/util/update-gym-fish-with-mirror.sh \
    && echo "cd /root/gym-fish" >> /root/util/update-gym-fish-with-mirror.sh \
    && echo "git remote set-url origin https://hub.fastgit.org/dongfangliu/gym-fish.git" >> /root/util/update-gym-fish-with-mirror.sh \
    && echo "git pull" >> /root/util/update-gym-fish-with-mirror.sh \
    && echo "git remote set-url origin https://www.github.com/dongfangliu/gym-fish.git" >> /root/util/update-gym-fish-with-mirror.sh

# Start & Stop Jupyter Notebook
RUN echo "#!/bin/sh" > /root/util/start-jupyter-notebook.sh \ 
    && echo "source activate gym-fish" >> /root/util/start-jupyter-notebook.sh \
    && echo "pkill jupyter-notebook" >> /root/util/start-jupyter-notebook.sh \ 
    && echo "pkill Xvfb" >> /root/util/start-jupyter-notebook.sh \ 
    && echo "rm -f /tmp/.X99-lock" >> /root/util/start-jupyter-notebook.sh \ 
    && echo "Xvfb :99 -screen 0 640x480x24 &" >> /root/util/start-jupyter-notebook.sh \ 
    && echo "jupyter-notebook --ip 0.0.0.0 --allow-root /root/" >> /root/util/start-jupyter-notebook.sh
RUN echo "#!/bin/sh" > /root/util/start-jupyter-notebook-in-background.sh \ 
    && echo "source activate gym-fish" >> /root/util/start-jupyter-notebook-in-background.sh \
    && echo "pkill jupyter-notebook" >> /root/util/start-jupyter-notebook-in-background.sh \ 
    && echo "pkill Xvfb" >> /root/util/start-jupyter-notebook-in-background.sh \ 
    && echo "rm -f /tmp/.X99-lock" >> /root/util/start-jupyter-notebook-in-background.sh \ 
    && echo "Xvfb :99 -screen 0 640x480x24 &" >> /root/util/start-jupyter-notebook-in-background.sh \ 
    && echo "jupyter-notebook --ip 0.0.0.0 --allow-root /root/ 2&>1 >/dev/null &" >> /root/util/start-jupyter-notebook-in-background.sh
RUN echo "#!/bin/sh" > /root/util/stop-jupyter-notebook.sh \ 
    && echo "pkill jupyter-notebook" >> /root/util/stop-jupyter-notebook.sh \
    && echo "pkill Xvfb" >> /root/util/stop-jupyter-notebook.sh \ 
    && echo "rm -f /tmp/.X99-lock" >> /root/util/stop-jupyter-notebook.sh

# Start & Stop virtual Display
RUN echo "#!/bin/sh" > /root/util/start-virtual-display.sh \ 
    && echo "pkill Xvfb" >> /root/util/start-virtual-display.sh \ 
    && echo "rm -f /tmp/.X99-lock" >> /root/util/start-virtual-display.sh \
    && echo "Xvfb :99 -screen 0 640x480x24 &" >> /root/util/start-virtual-display.sh
RUN echo "#!/bin/sh" > /root/util/stop-virtual-display.sh \ 
    && echo "pkill Xvfb" >> /root/util/stop-virtual-display.sh \
    && echo "rm -f /tmp/.X99-lock" >> /root/util/stop-virtual-display.sh

# Prepare Installing Scripts
RUN mkdir /root/util/installed 

## Install Base Utils (git wget pip xvfb gcc g++)
RUN echo "#!/bin/sh" > /root/util/installed/install-base-utils.sh \
    && echo "apt-get update -y" >> /root/util/installed/install-base-utils.sh \
    && echo "apt-get install -y --no-install-recommends software-properties-common git wget python3-pip xvfb gcc g++" >> /root/util/installed/install-base-utils.sh

## Install Libdart
RUN echo "#!/bin/sh" > /root/util/installed/install-libdart.sh \
    && echo "apt-add-repository ppa:dartsim/ppa && apt-get update -y" >> /root/util/installed/install-libdart.sh \
    && echo "apt-get install -y --no-install-recommends libdart6=6.11.1-2016~202110221940~ubuntu18.04.1" >> /root/util/installed/install-libdart.sh
    
## Install Miniconda
RUN echo "#!/bin/sh" > /root/util/installed/install-conda.sh \
    && echo "wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'" >> /root/util/installed/install-conda.sh \
    && echo "sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda" >> /root/util/installed/install-conda.sh \
    && echo "rm Miniconda3-latest-Linux-x86_64.sh" >> /root/util/installed/install-conda.sh \
    && echo "ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh" >> /root/util/installed/install-conda.sh
    
## Install gym-fish python dependecies
RUN echo "#!/bin/sh" > /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "/opt/conda/bin/conda create -n gym-fish python=3.6.9 -y" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "/opt/conda/bin/conda init bash && . /root/.bashrc" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "source activate gym-fish" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "mkdir -p \${CONDA_PREFIX}/etc/conda/activate.d" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "mkdir -p \${CONDA_PREFIX}/etc/conda/deactivate.d" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "touch \${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "touch \${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo 'export OLD_LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}' >> \${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo 'export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}' >> \${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo 'export LD_LIBRARY_PATH=\${OLD_LD_LIBRARY_PATH}' >> \${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo 'unset OLD_LD_LIBRARY_PATH' >> \${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "cd /root/gym-fish" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "pip install -e ." >> /root/util/installed/install-conda-env-for-gym-fish.sh \ 
    && echo "source deactivate gym-fish" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo '# Activate gym-fish conda environment automatically' >> /root/.bashrc" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo 'source activate gym-fish' >> /root/.bashrc" >> /root/util/installed/install-conda-env-for-gym-fish.sh \
    && echo "echo '' >> /root/.bashrc" >> /root/util/installed/install-conda-env-for-gym-fish.sh

## Install jupyter notebook
RUN echo "#!/bin/sh" > /root/util/installed/install-jupyter-for-gym-fish.sh \
    && echo "source activate gym-fish" >> /root/util/installed/install-jupyter-for-gym-fish.sh \
    && echo "/opt/conda/bin/conda install jupyter notebook" >> /root/util/installed/install-jupyter-for-gym-fish.sh \
    && echo "source deactivate gym-fish" >> /root/util/installed/install-jupyter-for-gym-fish.sh

## Install gym-fish
RUN echo "#!/bin/sh" > /root/util/installed/install-gym-fish-from-mirror.sh \
    && echo "git clone https://hub.fastgit.org/dongfangliu/gym-fish.git /root/gym-fish" >> /root/util/installed/install-gym-fish-from-mirror.sh \
    && echo "cd /root/gym-fish" >> /root/util/installed/install-gym-fish-from-mirror.sh \
    && echo "git remote set-url origin https://github.com/dongfangliu/gym-fish.git" >> /root/util/installed/install-gym-fish-from-mirror.sh
RUN echo "#!/bin/sh" > /root/util/installed/install-gym-fish.sh \
    && echo "git clone https://github.com/dongfangliu/gym-fish.git /root/gym-fish" >> /root/util/installed/install-gym-fish.sh

## Install Scripts
RUN echo "#!/bin/sh" > /root/util/installed/install.sh \    
    && echo ". /root/util/installed/install-base-utils.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-libdart.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-conda.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-gym-fish.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-gym-fish-from-mirror.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-conda-env-for-gym-fish.sh" >> /root/util/installed/install.sh \
    && echo ". /root/util/installed/install-jupyter-for-gym-fish.sh" >> /root/util/installed/install.sh

# Configure Scripts and Conda
RUN chmod +x -R /root/util/
ENV PATH=${PATH}:/root/util:/opt/conda/bin

# Main Docker Build
RUN . /root/util/set-mirrors.sh \
    && /bin/bash /root/util/installed/install.sh \
    && . /root/util/unset-mirrors.sh \
    && apt-get autoremove -y gcc g++ wget \
    && apt-get autoremove -y \
    && /opt/conda/bin/conda clean -a -f -y \
    && rm -rf /root/.cache/* \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Welcome info 
RUN echo "# Welcome info" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc \
    && echo "echo 'Welcome to gym-fish docker'" >> /root/.bashrc \
    && echo "echo 'We have configured most of the requirements here to start gym-fish demos'" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc \
    && echo "echo 'To run a basic environment test:'" >> /root/.bashrc \
    && echo "echo '    sh /root/util/start-virtual-display.sh && python /root/gym-fish/quick_env_test.py'" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc \
    && echo "echo 'To start the jupyter notebook:'" >> /root/.bashrc \
    && echo "echo '    sh /root/util/start-jupyter-notebook.sh'" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc \
    && echo "echo 'Please refer to the document at https://gym-fish.readthedocs.io/ to train and run policy'" >> /root/.bashrc \
    && echo "echo 'For example:'" >> /root/.bashrc \
    && echo "echo '    python3 /root/gym-fish/train.py --env fish-pose-control-v0 --gpu-id 0 --n-timesteps 50000 --eval-freq 2000 --eval-episodes 1'" >> /root/.bashrc \
    && echo "echo '    python3 /root/gym-fish/enjoy.py --env koi-cruising-v0 --gpu-id 0'" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc \
    && echo "echo 'We have prepared some other useful scripts in /root/util:'" >> /root/.bashrc \
    && echo "ls /root/util" >> /root/.bashrc \
    && echo "echo ''" >> /root/.bashrc

# Update Gym-fish
RUN . /root/util/update-gym-fish-with-mirror.sh
