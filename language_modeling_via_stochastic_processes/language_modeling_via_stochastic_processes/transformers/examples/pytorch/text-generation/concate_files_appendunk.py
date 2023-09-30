import sys

# diffusion_max50_feb7_notuselatent_0_gen_
prefix = sys.argv[1]

lines = []

for sss in range(1, 6):
    filename = f'{prefix}{sss}.txt'
    lines.extend(open(filename).readlines())
    if sss == 1:
        a = len(lines)
lines.extend(['Unknown Title\n']*a)

with open(f'unk_concat_{prefix}.txt', 'w') as fout:
    for line in lines:
        fout.write(line)
