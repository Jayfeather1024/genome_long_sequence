import sys
from sacrebleu import corpus_bleu

output_lns = open(sys.argv[2]).readlines()
refs_lns = open(sys.argv[1]).readlines()

score = corpus_bleu(output_lns, [refs_lns])
print (score)


output_lns = open(sys.argv[2]).readlines()
refs_lns = open(sys.argv[1]).readlines()

def rm_func(lines_in):
    lines = []
    for line in lines_in:
        if 'Unknown Title\n' == line:
            continue
        lines.append(line)
    return lines
output_lns = rm_func(output_lns)
refs_lns = rm_func(refs_lns)

score = corpus_bleu(output_lns, [refs_lns])
print (score)
