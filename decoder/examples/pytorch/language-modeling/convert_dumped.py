import sys, os, torch
import numpy as np

data = torch.load('dumped_sentence_embeddings.pt')
data = torch.load('/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes//wikihow/encoder_93810/run_l0.0005_b32/diffusion/zs.pt')

for split in ['train', 'val', 'test']:
    ddd = data[split]
    fff = []
    for e in ddd:
        fff.append(len(e['sentence_embeddings']))
    fff = np.array(fff)
    print (split)
    print (f'mean: {fff.mean()}')
    print (f'min: {fff.min()}')
    if split == 'train':
        train_max = fff.max()
    print (f'max: {fff.max()}')
    print (f'median: {np.median(fff)}')

#for split in data:
#    ddd = data[split]
#    for e in ddd:
#        #import pdb; pdb.set_trace()
#        sentence_embeddings = e['sentence_embeddings']
#        last_embedding = sentence_embeddings[-1]
#        e['num_sentences'] = len(sentence_embeddings)
#        padded_sentence_embeddings = sentence_embeddings.new_zeros(train_max, sentence_embeddings.shape[-1]).fill_(0)
#        padded_sentence_embeddings[:e['num_sentences'], :] = sentence_embeddings
#        if e['num_sentences'] < train_max:
#            padded_sentence_embeddings[e['num_sentences']:, :] = last_embedding.view(1, -1).expand(train_max-e['num_sentences'], -1)
#        #import pdb; pdb.set_trace()
#
#        e['padded_sentence_embeddings'] = padded_sentence_embeddings
#torch.save(data, 'padded_sentence_embeddings.pt')
