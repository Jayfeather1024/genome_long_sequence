import random
random.seed(42)
import re
import matplotlib.pyplot as plt
import os

with open('test_srun.sh') as fin:
    template = fin.read()


for rounds in range(10):
    text = template
    seeds = []
    for i in range(6):
        rand = random.randint(10000, 99999)
        seeds.append(rand)

    #filename = f'encoder_{seeds[0]}/run_l0.0005_b32/diffusion/seed_{seeds[3]}/log.train_diffusion'
    #filename = f'encoder_28289/run_l0.0005_b32/diffusion/seed_81482/log.train_diffusion.{lr}'
    filename = f'log.train_diffusion.0.00005_{seeds[0]}'
    if not os.path.exists(filename):
        print (filename)
        continue
    with open(filename) as fin:
        eval_losses = []
        for line in fin:
            #wandb:    eval_loss 0.04946
            # | eval_loss    | 9.96     |  
            #m = re.match(r'wandb:\s+eval_loss\s([\.\d]+)', line.strip())
            m = re.match(r'.*?\|\s+loss\s+\|\s+([\.\d]+)\s+.*?', line.strip())
            if m:
                eval_loss = float(m.group(1))
                eval_losses.append(eval_loss)
            #print (filename, eval_loss)
        #import pdb; pdb.set_trace()
        #plt.figure()
        plt.plot(list(range(len(eval_losses))), eval_losses, label=f'{seeds[0]}')
        #plt.ylim(0, 0.1)
        plt.legend()
        plt.savefig(os.path.join('imgs_seeds', filename.replace('/', '_') + '.png'))

