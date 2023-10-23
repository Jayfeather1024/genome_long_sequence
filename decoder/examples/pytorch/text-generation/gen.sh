stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --no_eos --label=LM_wikisection_32 --seed=0 > log.gen 2>&1&
stdbuf -oL python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --no_eos --label=LM_wikisection_32 --seed=0 > log.gen.noeos.fixed 2>&1&

stdbuf -oL python run_decoding_from_embeddings_samplez0.py --model_type=gpt2 --model_name_or_path=../language-modeling/LM_wikisection_32/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection --encoder_filepath=../../../../outputs/2022-06-02/11-13-47/wikisection_tc32/lightning_logs/version_0/checkpoints/epoch=99-step=258199.ckpt --latent_dim=32 --project=LM_wikisection --no_eos --label=LM_wikisection_32 --seed=0 > log.gen.noeos.sampleb0.fixed 2>&1&