#!/bin/bash

# setting seed and file path
export SEED1=28289
export SEED2=23434
export SEED3=98696
export SEED_DIFFUSION=81482
export SEED_DIFFUSION2=21395
export SEED_DIFFUSION3=87397
export WANDB_API_KEY=f6021dca133c93e80a7dae4620bd335d4d08cac6
export BASE=/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_long_sequence/
export FILEBASE=/lus/eagle/projects/CVD-Mol-AI/for_yuntian/


#---------------------------------ENCODER TRAINING------------------------------
# activate environment
cd $BASE
source activate base
conda activate genome_infilling

# setting output file path and parameters for encoder
export ENCODER_BASE=${FILEBASE}/genome_results/encoder_genslm_${SEED1}_clean_code
mkdir -p ${ENCODER_BASE}
export BSZ=32; export LR=0.0005;
export ENCODER_B=${ENCODER_BASE}/run_l${LR}_b${BSZ}
mkdir -p ${ENCODER_B}

# start encoder training
echo "training encoder"
# CUDA_VISIBLE_DEVICES=0 python encoder/train_encoder.py --config-name=genslm experiment_params.seed=${SEED1} wandb_settings.exp_dir=${ENCODER_B}/ data_params.name=genome_sars model_params.latent_dim=32 optim_params.learning_rate=${LR} experiment_params.num_epochs=5 > ${ENCODER_B}/log.train.encoder.shuffle 2>&1


#---------------------------------DIFFUSION MODEL TRAINING------------------------------
# dumping then training diffusion (generate latent embeddings using encoder)
export DIFFUSION=${ENCODER_B}/diffusion
mkdir -p ${DIFFUSION}
cd ${BASE}/decoder/examples/pytorch/language-modeling
echo "dumping states"
# CUDA_VISIBLE_DEVICES=0 stdbuf -oL python run_time_clm_dumpstates_genslm.py --model_name_or_path genslm --dataset_name genome_sars --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=1 --seed=1 --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --output_dir tmp_dump --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 0 --fname ${DIFFUSION}/zs.pt > ${DIFFUSION}/log.dumpstates 2>&1

# start training diffusion model
export DIFFUSION_BASE=/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_long_sequence/Diffusion-LM/improved-diffusion
cd $DIFFUSION_BASE
export LR=0.00002
export DIFFUSION_B=${DIFFUSION}/seed_${SEED_DIFFUSION}_lr${LR}
mkdir -p ${DIFFUSION_B}
conda activate diffusion_lm
echo "training diffusion"
# CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=${DIFFUSION_B} TOKENIZERS_PARALLELISM=false python scripts/train.py --checkpoint_path ${DIFFUSION_B} --model_arch transformer --modality highlevel-tgt --save_interval 50000 --lr $LR --batch_size 64  --diffusion_steps 2000 --noise_schedule sqrt  --use_kl False --learn_sigma False  --image_size 8 --num_channels 128 --seed ${SEED_DIFFUSION} --dropout 0.1 --in_channel 32 --out_channel 32 --padding_mode block --experiment random  --lr_anneal_steps 200000 --weight_decay 0.0 --num_res_blocks 2  --predict_xstart True --training_mode highlevel --vocab_size 821 --highlevel_train ${DIFFUSION}/zs.pt --config_name bert-large-uncased > ${DIFFUSION_B}/log.train_diffusion.lr${LR}.bertlarge 2>&1


#---------------------------------DECODER TRAINING------------------------------
# setting output file path and parameters for decoder
conda activate genome_infilling
export DECODER_BASE=${ENCODER_B}/decoder_${SEED2}
mkdir -p ${DECODER_BASE}
export MODEL=genslm; export EPOCHS=5;
export GAUSSIAN=1.0
export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_d${GAUSSIAN}_nosmooth_gaussian
mkdir -p ${DECODER_B}
echo ${DECODER_B}
cd ${BASE}/decoder/examples/pytorch/language-modeling

# start decoder training
echo "training decoder $MODEL"
# CUDA_VISIBLE_DEVICES=0 stdbuf -oL python run_time_clm_genslm_chunk.py --model_name_or_path ${MODEL} --dataset_name genome_sars --encoder_mode=gaussian_${GAUSSIAN} --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=${EPOCHS} --seed=${SEED2} --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --output_dir ${DECODER_B} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 0 --dropout_diffusion 0  > ${DECODER_B}/log.train.decoder.d${GAUSSIAN} 2>&1


#---------------------------------NO LATENT DECODER BASELINE------------------------------
export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_not_use_latent_SARS
mkdir -p ${DECODER_B}
cd ${BASE}/decoder/examples/pytorch/language-modeling
echo "training no latent decoder $MODEL"
# CUDA_VISIBLE_DEVICES=0 stdbuf -oL python run_time_clm_genslm_chunk.py --model_name_or_path ${MODEL} --dataset_name genome_sars --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=${EPOCHS} --seed=${SEED2} --encoder_filepath=${ENCODER_B}/checkpoints/ --latent_dim=32 --output_dir ${DECODER_B} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings --not_use_latent 1 --encoder_mode=not_use_latent > ${DECODER_B}/log.train.decoder 2>&1


#---------------------------------DIFFUSION GENERATION------------------------------
conda activate diffusion_lm
cd $DIFFUSION_BASE
export DIFFUSION_O=${DIFFUSION_B}/sample_${SEED_DIFFUSION2}_repaint
mkdir -p ${DIFFUSION_O}
echo "generating diffusion"
for SAMPLESEED in 01234
do
    echo $SAMPLESEED
    # CUDA_VISIBLE_DEVICES=0 python scripts/text_sample_infill_repaint_genslm.py --model_path ${DIFFUSION_B}/ema_0.9999_000000.pt --batch_size 100 --num_samples 100 --top_p -1.0 --out_dir ${DIFFUSION_O} --clamp none --seed ${SAMPLESEED} --size 20 --highlevel_train ${DIFFUSION}/zs.pt  > ${DIFFUSION_O}/log.gen.seeds.$SAMPLESEED 2>&1
done

#---------------------------------DECODER INFILLING GENERATION------------------------------
cd $BASE
conda activate genome_infilling
cd ${BASE}/decoder/examples/pytorch/text-generation
export DECODER_B=${DECODER_BASE}/decoder_${MODEL}_e${EPOCHS}_d${GAUSSIAN}_nosmooth_gaussian

export DECODER_B_GEN=${DECODER_B}/gen_diffusion_${SEED_DIFFUSION3}
mkdir -p ${DECODER_B_GEN}
mkdir -p ${DECODER_B_GEN}/generations
mkdir -p ${DECODER_B_GEN}/generations_only_model
mkdir -p ${DECODER_B_GEN}/generations_no_latent
echo "generating diffusion for $MODEL"
# generation with high level vectors
# CUDA_VISIBLE_DEVICES=0 python generate_genslm_25M.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=genome_sars --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_genslm --label=LM_genslm_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.seeds.1234 --fname ${DECODER_B_GEN}/generations --method=beam --output_dir ${DECODER_B_GEN}/generations --oracle True --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.oracle.True 2>&1
# generation with genslm model
# CUDA_VISIBLE_DEVICES=0 python generate_genslm_25M_only_model.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=genome_sars --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_genslm --label=LM_genslm_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.seeds.1234 --fname ${DECODER_B_GEN}/generations_only_model --method=beam --output_dir ${DECODER_B_GEN}/generations_only_model --oracle False --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.onlymodel 2>&1
# generation with no latent decoder
# CUDA_VISIBLE_DEVICES=0 python generate_genslm_25M_no_latent.py --model_type=${MODEL} --model_name_or_path=${DECODER_B} --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=genome_sars --encoder_filepath=${ENCODER_B}/checkpoints --latent_dim=32 --project=LM_genslm --label=LM_genslm_32_eos --seed=${SEED_DIFFUSION3} --num_return_sequences=20 --num_intervals_factor 1 --sample_file ${DIFFUSION_O}/zs.pt.masked.seeds.1234 --fname ${DECODER_B_GEN}/generations_no_latent --method=beam --output_dir ${DECODER_B_GEN}/generations_no_latent --oracle False --encoder_mode=gaussian_0 > ${DECODER_B_GEN}/log.nolatent 2>&1