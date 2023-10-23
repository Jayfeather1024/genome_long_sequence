import sys
from sacrebleu import corpus_bleu

from  summ_eval.bleu_metric import BleuMetric

output_lns = open(sys.argv[2]).readlines()
refs_lns = open(sys.argv[1]).readlines()

#score = corpus_bleu(output_lns, [refs_lns])
#print (score)
#
#
#output_lns = open(sys.argv[2]).readlines()
#refs_lns = open(sys.argv[1]).readlines()

def rm_func(lines_in):
    lines = []
    for line in lines_in:
        if 'Unknown Title\n' == line:
            continue
        #line = line.replace('<|endoftext|.', '').strip()+'\n'
        #if '!' in line:
        #    #import pdb; pdb.set_trace()
        #    idx = line.index('!')
        #    line = line[:idx+1] + '\n'
        lines.append(line)
    return lines
output_lns = rm_func(output_lns)
refs_lns = rm_func(refs_lns)

scores = []
#import pdb; pdb.set_trace()
for line_sys, line_ref in zip(output_lns, refs_lns):
    line_sys = line_sys.strip()
    line_ref = line_ref.strip()
    score = BleuMetric().evaluate_batch([line_sys], [line_ref])
    #print (line_sys, score)
    score = score['bleu']
    scores.append(score)

print (sum(scores)/len(scores))

#score = corpus_bleu(output_lns, [refs_lns])
