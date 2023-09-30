python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow data_params.name=wikisection model_params.latent_dim=32
python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow data_params.name=wikisectioncbow model_params.latent_dim=32 > log.train_cbow 2>&1&
outputs/2022-06-21/14-34-48/wikisection_tc32_cbow_requires_grad_single/lightning_logs/version_0/


export LR=0.0001; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > logs/log.vanilla.lr$LR 2>&1&
export LR=0.0005; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > logs/log.vanilla.lr$LR 2>&1&

export LR=0.0001; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&
export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&
export LR=0.00005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&


DONE 11.69, 36.88 export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > rerun_logs/log.vanilla.lr$LR 2>&1&
/n/holylfs05/LABS/rush_lab/Lab/users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/outputs/2022-06-23/12-47-18/rerun_wikisection_tc32_cbow_0.0001/checkpoints/epoch=93-step=242707.ckpt


RUNNING export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > rerun_logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&


DONE 10.83 39.40 export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > rerun_logs/log.vanilla.lr$LR 2>&1&
DONE 11.59 41.47 export LR=0.001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > rerun_logs/log.vanilla.lr$LR 2>&1&
DONE 11.78 35.19 export LR=0.005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=$LR > rerun_logs/log.vanilla.lr$LR 2>&1&


RUNNING export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > rerun_logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&
DELAY export LR=0.00005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > rerun_logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&



RUNNING export LR=0.001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > rerun_logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&
RUNNING export LR=0.005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_$LR data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR > rerun_logs/log.finetune_gpt2.single_layer.lr$LR 2>&1&





DONE 310.87 47.77 export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > rerun_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
DONE 1320.09 47.62 export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > rerun_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
DONE 1303.09 47.12 export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > rerun_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&

DONE 1047.95 47.13 export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > rerun_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
DONE 20.84 47.13 export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > rerun_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
DONE 10.83 39.40 export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=rerun_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > rerun_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&
