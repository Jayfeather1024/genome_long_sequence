import sys
import os
import csv
from transformers import AutoTokenizer
import torch
import copy

PATH2HUGGINGFACE='/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/' # CHANGE ME! 

def _check_ordering(raw_seq, section_names, dataset_name):
    #import pdb; pdb.set_trace()
    # Check ordering
    #if 'taskmaster' in self.dataset_name:
    #    return self._taskmaster_ordering(input_ids, raw_seq, info)
    section_order_idxs = []
    correct_ordering = True
    for section_name in section_names:
        if section_name in raw_seq:
            idxs = [i for i in range(len(raw_seq))
                    if raw_seq.startswith(section_name, i)]
            if len(idxs) > 1:
                if 'wikihow' in dataset_name and 'STEP' in section_name:
                    #### Get step numbers
                    ###step_num = [raw_seq[i:].split(' ')[4] for i in idxs]
                    try:
                        ###step_num = [raw_seq[i:].split(' ')[4] for i in idxs]
                        step_num = [raw_seq[i:].split(' ')[3] for i in idxs]
                    # intify:
                    #try:
                        step_num = [int(i) for i in step_num]
                        #correct_ordering = all([step_num[i] < step_num[i+1] for i in range(len(step_num)-1)])
                        correct_ordering = all([step_num[i] == i for i in range(len(step_num))])
                    except:
                        correct_ordering = False # not an int following STEP
                else:
                    correct_ordering = False
            section_order_idxs += idxs
        else:
            correct_ordering = False

    for i in range(len(section_order_idxs)):
        for j in range(len(section_order_idxs)):
            if j <= i: continue
            correct_ordering = (
                correct_ordering and
                section_order_idxs[i] < section_order_idxs[j])
    return correct_ordering

def get_input_ids(text, tokenizer):
    input_ids = torch.LongTensor(tokenizer.encode(text))
    return input_ids

def check_present(text, section_names):
    present = {}
    for section_name in section_names:
        if section_name in text:
            present[section_name] = True
        else:
            present[section_name] = False
    return present

def check_redundancy(text, section_names):
    redundant = {}
    for section_name in section_names:
        cnt = text.count(section_name)
        if cnt > 1:
            flag = True
        else:
            flag = False
        redundant[section_name] = flag
    return redundant


def check_section_length(input_ids, section_ids, section_names):
    lengths = {}
    #import pdb; pdb.set_trace()
    for section_id, section_name in zip(section_ids, section_names):
        idxs = (input_ids == section_id).nonzero(as_tuple=True)
        length = 0
        for start_idx in idxs[0]:
            start_idx = start_idx.item()
            # Check for next section id
            text_rest = input_ids[start_idx+1:].tolist() # +1 because you want to skip current token
            other_idxs = [start_idx+1+idx for idx, token in enumerate(text_rest)
                          if token in section_ids]
            if other_idxs: # non empty list
                end_idx = min(other_idxs)
            else: # last section
                end_idx = input_ids.shape[-1]
            length += end_idx - start_idx
        lengths[section_name] = length
    return lengths

#,total length,ordering,[ TITLE ] present,[ TITLE ] redundant,[ TITLE ] length,[ METHOD ] present,[ METHOD ] redundant,[ METHOD ] length,[ STEP ] present,[ STEP ] redundant,[ STEP ] length
def get_statistics(csv_filename, csv_out_filename, dataset_name):
    if "wikisection" == dataset_name:
        SECTION_IDS = ['[ ABSTRACT ]', '[ HISTORY ]', '[ GEOGRAPHY ]', '[ DEMOGRAPHICS ]']
    if 'recipe' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ INGREDIENTS ]',
            '[ DIRECTIONS ]'
        ]
    if 'tm2' in dataset_name or 'tickettalk' in dataset_name:
        SECTION_IDS = [
            '[ USER ]',
            '[ ASSISTANT ]',
        ]
    if 'wikihow' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ METHOD ]',
            '[ STEP ]'
        ]
    section_names = SECTION_IDS
    text_id = None
    row_out = []
    tokenizer_kwargs = {
        "cache_dir": PATH2HUGGINGFACE,
        "use_fast": True,
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained('gpt2', **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    #import pdb; pdb.set_trace()
    eos = tokenizer(' . ')['input_ids']
    print("Old tokenizer size: ", len(tokenizer))
    SECTION_IDS = copy.deepcopy(SECTION_IDS)
    SECTION_IDS += [' . ']
    if len(eos) == 1 and eos[0] == 50256 + len(SECTION_IDS):
        print("Not adding because it's already contained")
    else:
        print("Adding tokens, ", SECTION_IDS)
        tokenizer.add_tokens(SECTION_IDS)
    print("New tokenizer size: ", len(tokenizer))
    SPECIAL_TOKENS = [_[0] for _ in tokenizer(SECTION_IDS)['input_ids']]
    section_ids = SPECIAL_TOKENS[:-1]
    #import pdb; pdb.set_trace()
    with open(csv_filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        with open(csv_out_filename, 'w', newline='') as csv_outfile:
            spamwriter = csv.writer(csv_outfile, delimiter=',')
            row = ['', 'total length', 'ordering']
            for section_name in section_names:
                row.append(f'[ {section_name} ] present')
                row.append(f'[ {section_name} ] redundant')
                row.append(f'[ {section_name} ] length')
            spamwriter.writerow(row)

            idx = -1
            for row in spamreader:
                if text_id is None:
                    for id, e in enumerate(row):
                        if e == 'text':
                            text_id = id
                else:
                    idx += 1
                    text = row[text_id]
                    text_tok = text + "<|endoftext|>"
                    text_tok = text_tok.replace('. ', ' . ')
                    #import pdb; pdb.set_trace()
                    input_ids = get_input_ids(text_tok, tokenizer)
                    ordering = _check_ordering(text, section_names, dataset_name)
                    total_length = len(input_ids)
                    present = check_present(text, section_names)
                    redundant = check_redundancy(text, section_names)
                    lengths = check_section_length(input_ids, section_ids, section_names)
                    row = [f'{idx}', f'{total_length}', ordering]
                    for section_name in section_names:
                        row.append(f'{present[section_name]}')
                        row.append(f'{redundant[section_name]}')
                        row.append(f'{lengths[section_name]}')
                    spamwriter.writerow(row)




get_statistics('debug_fixed_trueCLEmbs_120_examples.csv', 'fixed_metrics.csv', 'wikihow')
