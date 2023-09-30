5.96 46.6 5.23 export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
RUNNING 5.19 export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
5.93 46.6 5.24 export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
5.56 46.7 5.20 export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&
5.80 46.5 5.19 export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ > fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ} 2>&1&


7.57 36.79 5.18 export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > fixval_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&
7.00 39.40 5.08 export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > fixval_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&
6.84 40.42 5.07 export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > fixval_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&
6.75 39.61 5.13 export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > fixval_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&
8.19 35.19 5.34 export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=fixval_wikisection_tc32_cbow_${LR}_${BSZ} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} > fixval_logs/log.vanilla.lr${LR}.bsz${BSZ} 2>&1&



7.12 39.37 5.23 export LOG=False; export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
7.13 38.11 export LOG=True; export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
EVAL 6.96 40.10 5.08 export LOG=False; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
6.98 38.05 export LOG=True; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


EVAL 6.84 40.22 5.08 export LOG=False; export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
6.75 40.61 export LOG=True; export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
EVAL 6.66 41.23 5.12 export LOG=False; export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
6.66 41.15 export LOG=True; export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


EVAL 6.88 39.74 5.64 export LOG=False; export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
6.66 41.39 export LOG=True; export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 optim_params.learning_rate=${LR} loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.vanilla.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


export LOG=False; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&

export LOG=True; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&

export LOG=False; export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
 export LOG=True2; export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
export LOG=True2; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_single_layer_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=True model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.single_layer.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


5.27 export LOG=False; export BSZ=32; export LR=0.0001; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=False model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


5.23 export LOG=False; export BSZ=32; export LR=0.0005; CUDA_VISIBLE_DEVICES=0 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=False model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


5.24 export LOG=False; export BSZ=32; export LR=0.001; CUDA_VISIBLE_DEVICES=1 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=False model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&


5.24 export LOG=False; export BSZ=32; export LR=0.002; CUDA_VISIBLE_DEVICES=2 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=False model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&

5.50 export LOG=False; export BSZ=32; export LR=0.005; CUDA_VISIBLE_DEVICES=3 python scripts/train_encoder.py --config-name=cbow wandb_settings.exp_dir=2fixval_wikisection_tc32_cbow_finetune_gpt2_${LR}_${BSZ}_${LOG} data_params.name=wikisectioncbow model_params.latent_dim=32 model_params.single_layer=False model_params.finetune_gpt2=True optim_params.learning_rate=$LR optim_params.batch_size=$BSZ loss_params.loss=cbow2 model_params.use_log=${LOG} > 2fixval_logs/log.finetune_gpt2.lr${LR}.bsz${BSZ}.log${LOG} 2>&1&
