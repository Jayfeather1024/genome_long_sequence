import argparse
import os
import torch
from encoder.src.models import language
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import numpy as np


def load_encoder(model_path):
    model = language.GPT2NeoXOUEncoder(
         hidden_dim=128,
         latent_dim=32,
         finetune_gpt2=True, single_layer=False)

    state_dict = torch.load(model_path)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if "model." in k:
            new_dict[k[6:]] = v
        else:
            if 'opt_mlp' in k or k == 'sigma' or k == 'log_sigma':
                continue
            new_dict[k] = v

    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_latent_feature(sequence, model, tokenizer, device='cuda'):
    model = model.to(device)
    sequence_tokenize = tokenizer(sequence, padding=True, return_tensors='pt')
    input_ids = sequence_tokenize['input_ids'].to(device)
    attention_mask = sequence_tokenize['attention_mask'].to(device)
    latent_feature = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    return latent_feature


def main(genome_file, model_path, tokenizer_path, sequence_split_length=512, output_file="./latent_features"):

    # read data
    with open(genome_file, "r") as f:
        genome_lines = f.readlines()
    genome_list=[]
    seperator = '[SEP]'
    for genome_line in genome_lines:
        genome_split = genome_line.strip().split()
        genome = []
        for i in range(len(genome_split)//sequence_split_length+1):
            genome.append(' '.join(genome_split[i*512:(i+1)*512]))
        for i in range(len(genome)):
            # [SEP] in prefix
            if i==(len(genome)-1):
                genome[i] = f'{seperator} {genome[i]} <|endoftext|>'
            else:
                genome[i] = f'{seperator} {genome[i]} '
        genome_list.append(genome)

    # load model
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_encoder(model_path)

    # get latent space
    latent_feature_list = []
    for genome in genome_list:
        genome_latent_feature_list = []
        for sequence in genome:
            latent_feature = get_latent_feature(sequence, model, tokenizer)
            genome_latent_feature_list.append(latent_feature.squeeze(0))
        latent_feature_list.append(torch.stack(genome_latent_feature_list).cpu().numpy)

    np.save(output_file, latent_feature_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--genome_file', dest='genome_file', action='store', required=True, help='file contains genome data', type=str)
    parser.add_argument('--model_path', dest='model_path', action='store', help='path for the encoder checkpoint', type=str, default="/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt")
    parser.add_argument('--tokenizer_path', dest='tokenizer_path', action='store', help='path for the tokenizer', type=str, default="/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json")
    parser.add_argument('--sequence_split_length', dest='sequence_split_length', action='store', help='split length', type=int, default=512)
    parser.add_argument('--output_file', dest='output_file', action='store', help='file saving the latent features', type=str, default="latent_features.npy")
    
    args = parser.parse_args()
    args = vars(args)
    main(**args)