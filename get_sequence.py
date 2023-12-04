import argparse
import os
import torch
from encoder.src.models import language
from transformers import PreTrainedTokenizerFast, AutoConfig
from tokenizers import Tokenizer
import numpy as np
from transformers import GPTNeoXChunkForCausalLM
import torch.nn as nn


def load_encoder(model_path):
    model = language.GPT2NeoXOUEncoder(
         hidden_dim=128,
         latent_dim=32,
         finetune_gpt2=True, single_layer=False)
    model.opt_mlp = nn.Sequential(*[nn.Linear(model.latent_dim*2, model.hidden_dim*4), nn.ReLU(), nn.Linear(model.hidden_dim*4, model.latent_dim)]) 

    state_dict = torch.load(model_path)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if "model." in k:
            new_dict[k[6:]] = v
        else:
            if k == 'sigma' or k == 'log_sigma':
                continue
            new_dict[k] = v

    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_decoder(encoder_path, model_path, tokenizer_path):

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # load model
    config = AutoConfig.from_pretrained('/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/neox_25,290,752.json')
    model = GPTNeoXChunkForCausalLM(config)
    a = torch.load('/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/patric_25m_epoch01-val_loss_0.57_bias_removed.pt')['state_dict']
    b = {}
    for k in a:
        b[k.replace('model.', '')] = a[k]
    model.load_state_dict(b,strict=False)
    model.resize_token_embeddings(len(tokenizer))

    encoder = load_encoder(encoder_path)

    model.cl_model = encoder
    proj_model = nn.Linear(32, model.gpt_neox.embed_in.weight.shape[1])
    model.proj_model = proj_model

    model.load_state_dict(torch.load(model_path))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return tokenizer, model


def generate_sequence(sequence, model, tokenizer, previous_features, current_features, device='cuda'):
    # setting special tokens
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token = "[UNK]"
    bos_id = tokenizer.encode(bos_token, add_special_tokens=False, return_tensors='pt').item()
    eos_id = tokenizer.encode(eos_token, add_special_tokens=False, return_tensors='pt').item()
    unk_id = tokenizer.encode(unk_token, add_special_tokens=False, return_tensors='pt').item()

    model = model.to(device)
    sequence = sequence + ' [SEP]'
    sequence_tokenize = tokenizer(sequence, padding=True, return_tensors='pt')
    input_ids = sequence_tokenize['input_ids'].to(device)
    attention_mask = sequence_tokenize['attention_mask'].to(device)

    sequence_length = input_ids.shape[-1]

    previous_total_features = previous_features.repeat(sequence_length-1, 1).to(device)
    current_features = current_features.to(device)
    previous_total_features = model.proj_model(previous_total_features)
    current_features = model.proj_model(current_features)

    model.past_seq_cl_feats = None
    model.chunk_ids = [0] * 1
    model.chunk_id = 0
    model.chunk_offset = 0
    
    sample_output = model.generate(input_ids, max_length=1025, min_length=1025,
            cl_feats=current_features,
            past_seq_cl_feats=previous_total_features,
            do_sample=True, top_p=0.9, eos_token_id=eos_id,
            bad_words_ids=[[unk_id]])

    sample = sample_output[0][sequence_length:]
    eos_pos = sample.eq(eos_id).nonzero().view(-1)
    if eos_pos.shape[0] > 0:
        eos_pos = eos_pos[0].item()
        sample = sample[:eos_pos+1]

    output_sequence = tokenizer.decode(sample, skip_special_tokens=False)
    output_sequence = output_sequence.replace('<|endoftext|>', '').strip()

    return output_sequence


def main(genome_file, feature_file, encoder_path, decoder_path, tokenizer_path):
    # load data
    with open(genome_file, "r") as f:
        genome_lines = f.readlines()
    genome_list=[]
    seperator = '[SEP]'
    for genome_line in genome_lines:
        genome_split = genome_line.strip().split()
        genome = []
        for i in range(len(genome_split)//512+1):
            genome.append(' '.join(genome_split[i*512:(i+1)*512]))
        genome_list.append(genome)

    # load features
    features = torch.load(feature_file)

    sequence = genome_list[0][0]
    previous_features = features[0]['padded_sentence_embeddings'][0]
    current_features = features[0]['padded_sentence_embeddings'][1]

    # load model
    tokenizer, model = load_decoder(encoder_path, decoder_path, tokenizer_path)

    output_sequence = generate_sequence(sequence, model, tokenizer, previous_features, current_features)
    print(output_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--genome_file', dest='genome_file', action='store', help='file contains genome data', type=str, default="val.txt.10")
    parser.add_argument('--feature_file', dest='feature_file', action='store', help='file contains feature data', type=str, default="/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/diffusion/seed_81482_lr0.00002/sample_21395_repaint/zs.pt.masked.seeds.1234")
    parser.add_argument('--encoder_path', dest='encoder_path', action='store', help='path for the encoder checkpoint', type=str, default="/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt")
    parser.add_argument('--decoder_path', dest='decoder_path', action='store', help='path for the decoder checkpoint', type=str, default="/lus/eagle/projects/CVD-Mol-AI/for_yuntian//genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_full/pytorch_model.bin")
    parser.add_argument('--tokenizer_path', dest='tokenizer_path', action='store', help='path for the tokenizer', type=str, default="/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json")
    
    args = parser.parse_args()
    args = vars(args)
    main(**args)