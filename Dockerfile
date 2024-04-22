FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive

# Add a /.local/bin and /.local/lib directories to allow editable python
# installs by any user
RUN mkdir -p -m 777 /.local/bin /.local/lib

# Install python and other useful programs
RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        glances \
        vim \
        tmux \
        curl && \
    apt clean

# Add the bashrc to start up the container correctly for local development
COPY docker/container_bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Install Python requirements using the compiled version of the requirements
RUN pip install --no-cache-dir -U pip setuptools \
        wheel \
        importlib-metadata \
        tqdm \
        h5py \
        pyarrow \
        joblib \
        click \
        colorlog \
        pandas \
        wandb
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
