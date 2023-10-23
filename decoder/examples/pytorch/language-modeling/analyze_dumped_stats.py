import sys, os, torch
import numpy as np

data = torch.load('dumped_sentence_embeddings.pt')

for split in ['train', 'val', 'test']:
    ddd = data[split]
    fff = []
    for e in ddd:
        fff.append(len(e['sentence_embeddings']))
    fff = np.array(fff)
    print (split)
    print (f'mean: {fff.mean()}')
    print (f'min: {fff.min()}')
    print (f'max: {fff.max()}')
    print (f'median: {np.median(fff)}')
