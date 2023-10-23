import subprocess



use_oracle = 'False'
use_oracle = 'True'
for mode in ['gaussian', 'dropout']:
#for mode in ['dropout', 'gaussian']:
    max_bleu = 0
    max_d = None
    #for d in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    #for d in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    #for d in [1.0, 0.9, 0.8, 0.7, 0.6]:
    #for d in [1.0, 1.5, 2.0, 2.5]:
    for d in [1.0, 2.0, 4.0, 8.0, 16.0]:
        #for use_oracle in ['False', 'True']:
        for use_oracle in ['False']:
            for steps in [2]: # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                bleus = []
                for k in range(1, 6):
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# no         cbow_diff, 0test and mean
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# no         cbow_diff, 0test
#                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# no         cbow_diff, mean
#                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# cb        ow_diff, 0test
#                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_0test/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# cb        ow, mean
#                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}_cbow_diff/gen_diffusion_87397_debug_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
# re        vised training (no cbow), base
                    #cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16_working/encoder_28289/run_l0.0005_b32/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16_working/encoder_28289/run_l0.0005_b32/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean_{steps}/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean_{steps}/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
                    cmd = f'python compute_bleu_reproduce.py /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gt_{k}.txt /n/holyscratch01/rush_lab/Users/yuntian/slurm_smaller_roc_stories_rerun_mar19_may16/encoder_28289/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_d{d}_nosmooth_{mode}/gen_diffusion_87397_debug_0test_and_mean/oracle{use_oracle}_beam1_max400_feb14_notuselatent_0_gen_{k}.txt'
                    output = subprocess.check_output(cmd, shell=True).strip()
                    bleu = float(output.decode('utf-8'))
                    bleus.append(bleu)
                    #print (k, bleu)
                avg_bleu = sum(bleus) / len(bleus)
                if avg_bleu > max_bleu:
                    max_bleu = avg_bleu
                    max_d = d
                print (steps, mode, d, avg_bleu)
            #print (mode, max_d, max_bleu)
