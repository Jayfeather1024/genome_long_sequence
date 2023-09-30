stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > log.gen.eos.fixed 2>&1&



stdbuf -oL python run_decoding_from_embeddings_samplez0.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > log.gen.eos_b0sample.fixed 2>&1&



stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > log.gen.eos.fixed 2>&1&





# no latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/not_use_latent_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.not_use_latent 2>&1&
# latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.use_latent 2>&1&


# no latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-medium --model_name_or_path=../language-modeling/medium_not_use_latent_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.not_use_latent.medium 2>&1&
# latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-medium --model_name_or_path=../language-modeling/medium_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.use_latent.medium 2>&1&



# no latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-large --model_name_or_path=../language-modeling/large_not_use_latent_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.not_use_latent.large 2>&1&
# latents
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-large --model_name_or_path=../language-modeling/large_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 > results_sun/log.gen.eos.use_latent.large 2>&1&



# fixed bridge
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 --bridge_version new > results_sun/log.gen.eos.use_latent.bridgenew 2>&1&
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-medium --model_name_or_path=../language-modeling/medium_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 --bridge_version new > results_sun/log.gen.eos.use_latent.medium.bridgenew 2>&1&
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2-large --model_name_or_path=../language-modeling/large_LM_wikisection_32_rerun/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --label=LM_wikisection_32_eos --seed=0 --bridge_version new > results_sun/log.gen.eos.use_latent.large.bridgenew 2>&1&
