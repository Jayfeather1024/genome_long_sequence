#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

"""

import sys
import argparse
import logging

import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import tqdm

from transformers import (
    GPT2TimeLMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    GPTNeoXTimeForCausalLM,
    GPTNeoXChunkForCausalLM,
)

from encoder.src import constants

from generation_metrics import GenerationMetrics
import sys
sys.path.append('../language-modeling')
from run_time_clm_genslm_chunk import (
    get_checkpoint,
    get_special_tokens,
    get_data_paths)

from run_time_clm_dumpstates_genslm import get_dataset

from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast, AutoModel
from tokenizers import Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2TimeLMHeadModel, GPT2Tokenizer),
    "gpt2-medium": (GPT2TimeLMHeadModel, GPT2Tokenizer),
    "gpt2-large": (GPT2TimeLMHeadModel, GPT2Tokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def simulate_brownian_bridge(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    #if isinstance(B_0, torch.Tensor):
    #    B_0 = B_0.cpu().detach().numpy()
    #if isinstance(B_T, torch.Tensor):
    #    B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    bsz = B_0.shape[0]
    dim = B_0.shape[-1]
    #x_t = np.copy(B_0)
    x_t = B_0.data.clone()
    for step in range(num_samples - 2): # number of sentences
        #noise = np.sqrt(dt)*sigma*np.random.normal(mu, sigma, dim)
        noise = np.sqrt(dt)*sigma*(mu + sigma*torch.randn(bsz, dim).cuda())
        t = step/num_samples
        x_tp1 = x_t * (1- dt/(1-t)) + (dt/(1-t))*B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_tp1 = x_t

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return torch.stack(bridge, dim=0).transpose(0,1).contiguous()
def simulate_brownian_bridge2(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    #if isinstance(B_0, torch.Tensor):
    #    B_0 = B_0.cpu().detach().numpy()
    #if isinstance(B_T, torch.Tensor):
    #    B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    bsz = B_0.shape[0]
    dim = B_0.shape[-1]
    #x_t = np.copy(B_0)
    x_t = B_0.data.clone()
    for step in range(num_samples - 2): # number of sentences
        dim = B_0.shape[-1]
        #noise = np.sqrt(dt)*sigma*np.random.normal(mu, sigma, dim)
        noise = np.sqrt(dt)*sigma*(mu + sigma*torch.randn(bsz, dim).cuda())
        t = step/num_samples
        x_tp1 = x_t * (1- dt/(1-t)) + (dt/(1-t))*B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_t = x_tp1

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return torch.stack(bridge, dim=0).transpose(0,1).contiguous()

def split_text(raw_text):
    split_pattern = ". "
    split_raw_text = [_ + split_pattern for _ in raw_text.split(split_pattern)]
    split_raw_text[-1] = split_raw_text[-1].rstrip(split_pattern)
    return split_raw_text

def get_density(dataset, lm, cl_model):
    """Estimate density of last latent"""
    first_latents = []
    last_latents = []
    length = len(dataset)
    for text_i in range(length):
        first_latents.append(dataset.cl_embeddings[text_i][0].detach().cpu().numpy())
        last_latents.append(dataset.cl_embeddings[text_i][-1].detach().cpu().numpy())
    first_latents = np.array(first_latents)
    last_latents = np.array(last_latents)
    return first_latents.mean(0), first_latents.std(0), last_latents.mean(0), last_latents.std(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--encoder_mode",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--fname",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--oracle",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--num-sentences", type=int, default=0)
    parser.add_argument("--split-sentences", type=int, default=1)
    parser.add_argument("--multiply-sentences", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.99)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--sample_file", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")

    parser.add_argument("--no_eos", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--suppress_eos", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dataset_name", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--fixed_prompt", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--num_intervals", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--num_intervals_factor", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--use_dataset", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--project", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--encoder_filepath", type=str, required=True,default="", help="Text added prior to input.")
    parser.add_argument("--latent_dim", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--use_random_embs", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_true_end_latent", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--label", type=str, default="", help="Text added prior to input.")

    parser.add_argument("--method", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--bridge_version", type=str, default="old", help="Text added prior to input.")
    parser.add_argument("--first_sentence", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--full_section", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--autoregressive", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.use_section_null = 0

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)


    config = AutoConfig.from_pretrained('/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/neox_25,290,752.json')
    tokenizer_path = '/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/codon_wordlevel_100vocab_added.json'
    print (f'Loading tokenizer from {tokenizer_path}')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print (f'New vocab size: {len(tokenizer)}')
    SECTION_IDS = [' . ']
    SPECIAL_TOKENS = [90]
    # SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
    #     dataset_name=args.dataset_name, tokenizer=tokenizer)
    # -1 because of the added " . "
    config.max_num_sections = 0


    model = GPTNeoXChunkForCausalLM(config)
    a = torch.load('/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/patric_25m_epoch01-val_loss_0.57_bias_removed.pt')['state_dict']
    b = {}
    for k in a:
        #b[k.replace('module.model.', '')] = a[k]
        b[k.replace('model.', '')] = a[k]
    model.load_state_dict(b,strict=False)
    model.resize_token_embeddings(len(tokenizer))


    if os.path.isdir(args.encoder_filepath):
        import glob
        files = glob.glob(os.path.join(args.encoder_filepath, '*.ckpt'))
        assert len(files) == 1, os.path.join(args.encoder_filepath, '*.ckpt')
        args.encoder_filepath = files[0]
        print (args.encoder_filepath)

    cpu_device = torch.device('cpu')
    base_model = 'gpt2neox'
    CL_MODEL = get_checkpoint(
        dataset_name=args.dataset_name,
        latent_dim=args.latent_dim,
        sec_id=True,
        token_size= len(tokenizer),
        base_model=base_model,
        filepath=args.encoder_filepath
    )# .to(cpu_device)

    CL_MODEL.to(args.device)
    CL_MODEL.eval()

    model.cl_model = CL_MODEL
    proj_model = nn.Linear(32, model.gpt_neox.embed_in.weight.shape[1])
    model.proj_model = proj_model

    print(args.model_name_or_path)
    print (f'Loading model {os.path.join(args.model_name_or_path, "pytorch_model.bin")}')
    model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')))

    model.to(args.device)
    model.eval()


    fname = args.model_name_or_path.split('/')[-2]
    fname = args.fname
    args.encoder_type = 'contrastt'

    gt_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                                fname=fname+"_trueCLEmbs_" + str(args.num_intervals_factor) + str(args.num_return_sequences),
                                model_args=args,
                                subclass="GT")
    random_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                            model_args=args,
                                fname=fname+"_randomCLEmbs_"+ str(args.num_intervals_factor) + str(args.num_return_sequences),
                                subclass="RANDOM")
    bridge_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                                fname=fname+"_bridgeCLEmbs_"+ str(args.num_intervals_factor) + str(args.num_return_sequences),
                            model_args=args,
                                subclass="BRIDGE")

    if args.fp16:
        model.half()

    if args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'

    os.environ['WANDB_CONSOLE']='wrap'
    wandb.init(project=args.project)
    wandb.config.update(args)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    # model.transformer._config.use_noisy_embeddings = False
    logger.info(args)

    prompt_text = args.prompt if args.prompt else "" # input("Model prompt >>> ")


    # Data
    assert args.dataset_name
    print(f'Args: {args}')
    train_path, _, eval_path = get_data_paths(args)
    # train_path = "/home/jayfeather/language_modeling_via_stochastic_processes/data/roc_stories_genslm_toy/train.pkl"
    # eval_path = "/home/jayfeather/language_modeling_via_stochastic_processes/data/roc_stories_genslm_toy/test.pkl"

    #train_dataset = get_dataset(
    #    args=args,
    #    tokenizer=tokenizer,
    #    file_path=train_path,
    #    special_words=SECTION_IDS,
    #    cache_dir=constants.PATH2HUGGINGFACE,
    #    cl_model=CL_MODEL,
    #)
    eval_dataset = get_dataset(
        args=args,
        tokenizer=tokenizer,
        file_path=eval_path,
        special_words=SECTION_IDS,
        cache_dir=constants.PATH2HUGGINGFACE,
        cl_model=CL_MODEL,
    )

    num_intervals = len(eval_dataset)

    model.num_words = 0
    model.loss = 0
    data = torch.load(args.sample_file)
    #total_num_samples = data.shape[0]
    total_num_samples = len(data)
    total_num_samples = 100
    #total_num_samples = 10 # TODO: remove
    #for num_example in tqdm.tqdm(range(num_intervals * args.num_intervals_factor)):
    use_oracle = args.oracle == 'True'

    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token = "[UNK]"
    bos_id = tokenizer.encode(bos_token, add_special_tokens=False, return_tensors='pt').item()
    eos_id = tokenizer.encode(eos_token, add_special_tokens=False, return_tensors='pt').item()
    unk_id = tokenizer.encode(unk_token, add_special_tokens=False, return_tensors='pt').item()
    #import pdb; pdb.set_trace()
    for sss in range(1, 19):
        with open(os.path.join(args.output_dir, f'oracle{use_oracle}_beam1_max{total_num_samples}_gt_{sss}.txt'), 'w') as fgt:
            with open(os.path.join(args.output_dir, f'oracle{use_oracle}_beam1_max{total_num_samples}_gen_{sss}.txt'), 'w') as fgen:
                for num_example in tqdm.tqdm(range(total_num_samples)):
                    row = eval_dataset.cl_texts[num_example % len(eval_dataset)]
                    split_pattern = " [SEP] "
                    split_text = row.split(split_pattern)
                    split_text = [ _ + split_pattern for _ in split_text[:-1] ] + split_text[-1:]
                    split_text = split_text[0:-1]
                    # split_text[0] = "<|endoftext|> [SEP] " + split_text[0]
                    split_text[-1] = split_text[-1] + "<|endoftext|>"

                    fgt.write(f'{split_text[sss].replace(" [SEP] ", " ")}\n')

                    prefix = ''.join(split_text[sss-1])

                    total_cl_feats = data[num_example %len(eval_dataset)]['padded_sentence_embeddings'].cuda()
                    if args.encoder_mode.startswith('gaussian_'):
                        gaussian = float(args.encoder_mode.split('_')[-1])
                        total_cl_feats = total_cl_feats + gaussian * torch.randn(*total_cl_feats.shape).cuda()
                    total_cl_feats = model.proj_model(total_cl_feats)

                    if not use_oracle:
                        cl_feats = data[num_example %len(eval_dataset)][f'masked_{sss}'].cuda()
                        if args.encoder_mode.startswith('gaussian_'):
                            gaussian = float(args.encoder_mode.split('_')[-1])
                            cl_feats = cl_feats + gaussian * torch.randn(*cl_feats.shape).cuda()
                        cl_feats = model.proj_model(cl_feats)
                    else:
                        cl_feats = total_cl_feats[sss]

                    input_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').cuda()
                    prefix_length = input_ids.shape[-1]
                    input_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').cuda()
                    input_ids = input_ids.view(1, -1).expand(1, -1)
                    past_seq_cl_feats = total_cl_feats[sss-1].repeat(prefix_length-1, 1)

                    model.past_seq_cl_feats = None
                    model.chunk_ids = [0] * 1
                    model.chunk_id = 0
                    model.chunk_offset = 0
                    sample_output = model.generate(input_ids, max_length=1025, min_length=1025, \
                            cl_feats=cl_feats,
                            past_seq_cl_feats=past_seq_cl_feats,
                            do_sample=True, top_p=0.9, eos_token_id=eos_id, \
                            bad_words_ids=[[unk_id]])

                    sequences = []
                    #import pdb; pdb.set_trace()
                    for sample in sample_output:
                        sample = sample[prefix_length:]
                        eos_pos = sample.eq(eos_id).nonzero().view(-1)
                        if eos_pos.shape[0] > 0:
                            eos_pos = eos_pos[0].item()
                            sample = sample[:eos_pos+1]

                        text = tokenizer.decode(sample, skip_special_tokens=False)
                        text = text.replace('<|endoftext|>', '').strip()
                        sequences.append(text)
                    fgen.write(f'{text.replace(" [SEP] ", " ")}\n')

if __name__ == "__main__":
    main()