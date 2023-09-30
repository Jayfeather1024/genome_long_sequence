#!/bin/bash

export SEED1=28289
export SEED2=23434
export SEED3=98696
export SEED_DIFFUSION=81482
export SEED_DIFFUSION2=21395
export SEED_DIFFUSION3=87397
export WANDB_API_KEY=f6021dca133c93e80a7dae4620bd335d4d08cac6

export BASE=/home/jayfeather/genome_long_sequence/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/
export FILEBASE=/home/jayfeather/genome_long_sequence/
cd $BASE
source activate base
conda activate lm_via_sp
which python

export ENCODER_BASE=${FILEBASE}/genome_results/encoder_genslm_${SEED1}
mkdir -p ${ENCODER_BASE}
export BSZ=32; export LR=0.0005;
export ENCODER_B=${ENCODER_BASE}/run_l${LR}_b${BSZ}_cbow_diff
mkdir -p ${ENCODER_B}

#ENCODER_B: /n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32_cbow_diff
echo "training encoder"
# CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow_diff_genslm experiment_params.seed=${SEED1} wandb_settings.exp_dir=${ENCODER_B}/ data_params.name=roc_stories_genslm model_params.latent_dim=32 optim_params.learning_rate=${LR} experiment_params.num_epochs=5 > ${ENCODER_B}/log.train.encoder.shuffle 2>&1

#TODO: epoch 30


# DECODER_BASE:/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/decoder_2345
export DECODER_BASE=${ENCODER_B}/decoder_${SEED2}
mkdir -p ${DECODER_BASE}
#TODO: epochs 30
###
export MODEL=gpt2; export EPOCHS=5;
export GAUSSIAN=1.0
export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_d${GAUSSIAN}_nosmooth_gaussian

#DECODER_B: /n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/decoder_2345/decoder_gpt2_e1
mkdir -p ${DECODER_B}
echo ${DECODER_B}
cd ${BASE}/transformers/examples/pytorch/language-modeling
echo "training decoder $MODEL"
# CUDA_VISIBLE_DEVICES=0 stdbuf -oL python run_time_clm_genslm.py --model_name_or_path ${MODEL} --dataset_name roc_stories --encoder_mode=gaussian_${GAUSSIAN} --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=${EPOCHS} --seed=${SEED2} --encoder_filepath=${ENCODER_B} --latent_dim=32 --output_dir ${DECODER_B} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 0 --dropout_diffusion 0  > ${DECODER_B}/log.train.decoder.d${GAUSSIAN} 2>&1


cd ${BASE}/transformers/examples/pytorch/text-generation
export DECODER_B_GEN=${DECODER_B}/gen_${SEED3}
# DECODER_B_GEN: /n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/decoder_2345/decoder_gpt2_e1/gen_3456
mkdir -p ${DECODER_B_GEN}
echo "generating from decoder $MODEL"
# CUDA_VISIBLE_DEVICES=1 stdbuf -oL python run_decoding_from_embeddings_batch_genslm.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=roc_stories --encoder_filepath=${ENCODER_B}/checkpoints/ --latent_dim=32 --project=LM_roc_stories --label=LM_roc_stories_32_eos --seed=${SEED3} --num_return_sequences=20 --num_intervals_factor 1 --fname ${DECODER_B_GEN}/generations > ${DECODER_B_GEN}/log.gen 2>&1


# NO LATENT DECODER AND GENERATION
# #DECODER_B: /n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/decoder_2345/decoder_gpt2_e1_not_use_latent
# export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_not_use_latent
# mkdir -p ${DECODER_B}
# cd ${BASE}/transformers/examples/pytorch/language-modeling
# echo "training no latent decoder $MODEL"
# #stdbuf -oL python run_time_clm.py --model_name_or_path ${MODEL} --dataset_name roc_stories --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=${EPOCHS} --seed=${SEED2} --encoder_filepath=${ENCODER_B}/checkpoints/ --latent_dim=32 --output_dir ${DECODER_B} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 1 > ${DECODER_B}/log.train.decoder 2>&1
# ###
# cd ${BASE}/transformers/examples/pytorch/text-generation
# #DECODER_B_GEN: /n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/decoder_2345/decoder_gpt2_e1_not_use_latent/gen_3456
# export DECODER_B_GEN=${DECODER_B}/gen_${SEED3}
# mkdir -p ${DECODER_B_GEN}
# echo "generating from no latent decoder $MODEL"
# #stdbuf -oL python run_decoding_from_embeddings_batch.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=roc_stories --encoder_filepath=${ENCODER_B}/checkpoints/ --latent_dim=32 --project=LM_roc_stories --label=LM_roc_stories_32_eos --seed=${SEED3} --num_return_sequences=20 --num_intervals_factor 1 --fname ${DECODER_B_GEN}/generations > ${DECODER_B_GEN}/log.gen 2>&1



export DIFFUSION=${ENCODER_B}/diffusion
#DIFFUSION:/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/slurm/encoder_1234/run_l0.0005_b32/diffusion
mkdir -p ${DIFFUSION}
#first dumping then training diffusion then sampling diffusion then running generation.
cd ${BASE}/transformers/examples/pytorch/language-modeling
echo "dumping states"
# CUDA_VISIBLE_DEVICES=1 stdbuf -oL python run_time_clm_dumpstates_genslm.py --model_name_or_path gpt2 --dataset_name roc_stories --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=1 --seed=1 --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --output_dir tmp_dump --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 0 --fname ${DIFFUSION}/zs.pt > ${DIFFUSION}/log.dumpstates 2>&1


export DIFFUSION_BASE=/home/jayfeather/genome_long_sequence/language_modeling_via_stochastic_processes/Diffusion-LM/improved-diffusion
cd $DIFFUSION_BASE
export LR=0.00002
export DIFFUSION_B=${DIFFUSION}/seed_${SEED_DIFFUSION}_lr${LR}
mkdir -p ${DIFFUSION_B}
conda activate conda_diffusion
which python
echo "training diffusion"
# CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=${DIFFUSION_B} TOKENIZERS_PARALLELISM=false python scripts/train.py --checkpoint_path ${DIFFUSION_B} --model_arch transformer --modality highlevel-tgt --save_interval 50000 --lr $LR --batch_size 64  --diffusion_steps 2000 --noise_schedule sqrt  --use_kl False --learn_sigma False  --image_size 8 --num_channels 128 --seed ${SEED_DIFFUSION} --dropout 0.1 --in_channel 32 --out_channel 32 --padding_mode block --experiment random  --lr_anneal_steps 200000 --weight_decay 0.0 --num_res_blocks 2  --predict_xstart True --training_mode highlevel --vocab_size 821 --highlevel_train ${DIFFUSION}/zs.pt --config_name bert-large-uncased > ${DIFFUSION_B}/log.train_diffusion.lr${LR}.bertlarge 2>&1


# Generate from diffusion
export DIFFUSION_O=${DIFFUSION_B}/sample_${SEED_DIFFUSION2}_repaint
mkdir -p ${DIFFUSION_O}
echo "generating diffusion"
for SAMPLESEED in 01234
do
    echo $SAMPLESEED
    #CUDA_VISIBLE_DEVICES=0 python scripts/text_sample_infill_repaint_genslm.py --model_path ${DIFFUSION_B}/ema_0.9999_200000.pt --batch_size 100 --num_samples 100 --top_p -1.0 --out_dir ${DIFFUSION_O} --clamp none --seed ${SAMPLESEED} --size 20 --highlevel_train ${DIFFUSION}/zs.pt  > ${DIFFUSION_O}/log.gen.seeds.$SAMPLESEED 2>&1
done

# finally, generate with the diffusion process
cd $BASE
conda activate lm_via_sp
which python
cd ${BASE}/transformers/examples/pytorch/text-generation
export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_d${GAUSSIAN}_nosmooth_gaussian
export DECODER_B_GEN=${DECODER_B}/gen_diffusion_${SEED_DIFFUSION3}_new
mkdir -p ${DECODER_B_GEN}
mkdir -p ${DECODER_B_GEN}/generations
echo "generating diffusion for $MODEL"
# python run_decoding_from_embeddings_batch_full_roc_stories_smooth.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=roc_stories --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_roc_stories --label=LM_roc_stories_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.mean.seeds --fname ${DECODER_B_GEN}/generations_oracle --method=beam --output_dir ${DECODER_B_GEN}/generations_oracle --oracle True --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.oracle.True 2>&1
#CUDA_VISIBLE_DEVICES=0 python run_decoding_from_embeddings_batch_full_roc_stories_genslm.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=roc_stories --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_roc_stories --label=LM_roc_stories_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.seeds.1234 --fname ${DECODER_B_GEN}/generations --method=beam --output_dir ${DECODER_B_GEN}/generations --oracle False --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.oracle.False 2>&1
CUDA_VISIBLE_DEVICES=0 python generate_genslm_25M.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=roc_stories --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_roc_stories --label=LM_roc_stories_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.seeds.1234 --fname ${DECODER_B_GEN}/generations --method=beam --output_dir ${DECODER_B_GEN}/generations --oracle False --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.oracle.False 2>&1
