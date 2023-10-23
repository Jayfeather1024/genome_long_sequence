import re, sys, os, math
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import mauve

# not use latent
q1 = 'results_mon/not_use_latent_LM_wikisection_32_rerun_trueCLEmbs_sampleold120_examples.csv'
q2 = 'results_mon/medium_not_use_latent_LM_wikisection_32_rerun_trueCLEmbs_sampleold120_examples.csv'
q3 = 'results_mon/large_not_use_latent_LM_wikisection_32_rerun_trueCLEmbs_sampleold37_examples.csv'

# use gt latent
q4 = 'results_tue/LM_wikisection_32_rerun_trueCLEmbs_sampleold120_examples.csv'
q5 = 'results_tue/medium_LM_wikisection_32_rerun_trueCLEmbs_sampleold120_examples.csv'
q6 = 'results_tue/large_LM_wikisection_32_rerun_trueCLEmbs_sampleold37_examples.csv'

# use bridge latent old
q7 = 'results_tue/LM_wikisection_32_rerun_bridgeCLEmbs_sampleold120_examples.csv'
q8 = 'results_tue/medium_LM_wikisection_32_rerun_bridgeCLEmbs_sampleold120_examples.csv'
q9 = 'results_tue/large_LM_wikisection_32_rerun_bridgeCLEmbs_sampleold37_examples.csv'

# use bridge latent new
q10 = 'results_tue/LM_wikisection_32_rerun_bridgeCLEmbs_samplenew120_examples.csv'
q11 = 'results_tue/medium_LM_wikisection_32_rerun_bridgeCLEmbs_samplenew120_examples.csv'
q12 = 'results_tue/large_LM_wikisection_32_rerun_bridgeCLEmbs_samplenew37_examples.csv'

# use random latent
q13 = 'results_tue/LM_wikisection_32_rerun_randomCLEmbs_sampleold120_examples.csv'
q14 = 'results_tue/medium_LM_wikisection_32_rerun_randomCLEmbs_sampleold120_examples.csv'
q15 = 'results_tue/large_LM_wikisection_32_rerun_randomCLEmbs_sampleold37_examples.csv'

p = 'actual.txt'
p = 'val.txt'
p = 'train.txt'
p = 'test.txt'

def load_pfile(p_filename, max_num_lines=-1):
    outs = []
    with open(p_filename) as fin:
        for line in fin:
            line = line.replace('[ ACTUAL ]', '')
            line = line.replace('<|endoftext|>', '')
            line = line.strip()
            outs.append(line)
    if max_num_lines > 0:
        return outs[:max_num_lines]
    return outs

def load_file(q_filename, max_num_lines=-1):
    outs = []
    with open(q_filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in spamreader:
            i += 1
            if i == 1:
                continue
            line = row[1]
            line = line.replace('<|endoftext|>', '')
            line = line.strip()
            outs.append(line)
    if max_num_lines > 0:
        return outs[:max_num_lines]
    return outs

p_text = load_pfile(p, max_num_lines=-1)
for q_filename in [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]:
    q_text = load_file(q_filename, max_num_lines=-1)
    #import pdb; pdb.set_trace()
    
    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=1024, verbose=False)
    #import pdb; pdb.set_trace()
    print (q_filename)
    print (q_filename)
    print (len(p_text))
    print (len(q_text))
    print (out.mauve) # prints 0.9917
