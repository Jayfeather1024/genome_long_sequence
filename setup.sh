#!/bin/bash

# environment for encoder and decoder
conda create -n genome_infilling python=3.8.8
conda activate genome_infilling
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchmetrics==0.2.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
pip install -e ./ilm
pip install -e ./decoder

# environment for diffusion model
conda create -n diffusion_lm python=3.8
conda activate diffusion_lm
conda install mpi4py
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -e Diffusion-LM/improved-diffusion/ 
pip install -e Diffusion-LM/transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install blobfile==2.0.2
pip install wandb