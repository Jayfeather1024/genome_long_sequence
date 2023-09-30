stdbuf -oL python scripts/batch_decode.py diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e -1.0 ema > log.gen 2>&1&

python scripts/text_sample.py --model_path diffusion_models/diff_highlevel-tgt_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_highlevel/ema_0.9999_200000.pt --batch_size 50 --num_samples 1000 --top_p -1.0 --out_dir generation_outputs_highlevel --clamp none
