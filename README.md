# Setting Up Environment
Follow `setup.sh` step by step to create virtual environments. Two conda environments are needed, one for encoder and decoder training and the other for diffusion model training. 

# Dataset
Download the dataset into the folder `data`.

# Pretrained GenSLM
Download the pretrained GenSLM into the folder `genslm_model`.

# Getting the Latent Features
```
from get_latent_features import load_encoder, load_tokenizer, get_latent_feature

sequence = "ACG GGC CCG AGC AAA"
model_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt"
tokenizer_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json"

tokenizer = load_tokenizer(tokenizer_path)
model = load_encoder(model_path)
latent_feature = get_latent_feature(sequence, model, tokenizer)
print(latent_feature)
```
