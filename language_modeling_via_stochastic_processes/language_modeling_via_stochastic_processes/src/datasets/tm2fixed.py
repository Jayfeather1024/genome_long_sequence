from telnetlib import TM
import numpy as np
import os
import json
import random
import torch

from language_modeling_via_stochastic_processes.src.datasets import encoder
from language_modeling_via_stochastic_processes.src import constants

class TM2FixedDataset(encoder.BaseDataset):

    def __init__(
            self,
            train,
            tokenizer_name,
            seed,
            config,
            all_dataset=None):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )

    def _set_section_names(self):
        self.section_names = ['user', 'assistant']
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

        self.data_dir = constants.PATH2TM2
        if type(self.train) is str:
            print ('VAL DATASET')
            self.data_files = ['restaurant-search.json'] # 3276 conversations
            self.start_conversation = 0
            self.end_conversation = 300
        elif self.train:
            print ('TRAIN DATASET')
            self.data_files = ['restaurant-search.json'] # 3276 conversations
            self.start_conversation = 0
            self.start_conversation = 300
            self.end_conversation = 2000
        else:
            print ('TEST DATASET')
            self.data_files = ['restaurant-search.json']
            self.start_conversation = 2000
            self.end_conversation = 3276

    def _process_data(self):
        self.processed_data = []
        doc_id = 0
        min_length = np.inf
        for fname in self.data_files:
            data = json.load(open(os.path.join(self.data_dir, fname), 'rb'))
            data = data[self.start_conversation:self.end_conversation]
            print("num conversations loading ", len(data))
            for conversation in data:
                total_doc_sentences = len(conversation['utterances']) + 2
                #import pdb; pdb.set_trace()
                sentence_counter = 0
                sentence_info = {
                    "sentence": f'<|endoftext|> [ {conversation["utterances"][0]["speaker"].upper()} ]',
                    "sentence_id": sentence_counter,
                    "doc_id": doc_id,
                    "total_doc_sentences": total_doc_sentences
                }
                #doc_info.append(sentence_info)
                self.processed_data.append(sentence_info)
                sentence_counter += 1
                offset = 1
                for utterance in conversation['utterances']:
                    #text = "[ {} ] {}".format(utterance['speaker'].upper(), utterance['text'])
                    current_speaker = utterance['speaker'].upper()
                    if sentence_counter+1-offset < len(conversation['utterances']):
                        next_speaker = conversation['utterances'][sentence_counter+1-offset]['speaker'].upper()
                    else:
                        next_speaker = 'USER'
                    text = utterance['text']
                    #if sentence_counter == 0:
                    #    #text = f'<|endoftext|> [ {current_speaker} ] {text}'
                    #    text = f'<|endoftext|> [ {current_speaker} ] {text}'
                    #else:
                    #    text = f' {text}'
                    text = f' {text}'
                    if next_speaker is not None:
                        text = f'{text} [ {next_speaker} ]'
                    else:
                        assert False
                    #text = f"[ {utterance['speaker'].upper()} ] {utterance['text']}"
                    sentence_info = {
                        "sentence": text,
                        "sentence_id": sentence_counter,
                        "doc_id": doc_id,
                        "total_doc_sentences": total_doc_sentences
                    }
                    self.processed_data.append(sentence_info)
                    min_length = min(min_length, len(conversation['utterances']))
                    sentence_counter += 1
### <|endoftext|> [ USER ]| x y z. [ AST ]| a b c. [ USER ]| d e f. [ USER ]|<|endoftext|>
                sentence_info = {
                    "sentence": '<|endoftext|>',
                    "sentence_id": sentence_counter,
                    "doc_id": doc_id,
                    "total_doc_sentences": total_doc_sentences
                }
                #doc_info.append(sentence_info)
                self.processed_data.append(sentence_info)
                #import pdb; pdb.set_trace()
                sentence_counter += 1
###
                doc_id += 1
                if len(self.processed_data) > 25000:
                    break
            if len(self.processed_data) > 25000:
                break
        print(f'MIN LENGTH DISCOURSE: {min_length}')
        print("length of dataset: {}".format(len(self.processed_data)))
        print(f"last doc id {doc_id}")

class TM2FixedDiscourse(TM2FixedDataset):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        self.k = self.config.data_params.k

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order
        # Setup 2: sample t+k utterance
        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+self.k)
        t = max(0, tp1-self.k)

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if label: # in order
            pass # do nothing
        else:
            tmp = y_tp1
            y_tp1 = y_t
            y_t = tmp

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            't': t,
            'tp1': tp1,
            'label': label,
            'idx': index
        }
        return result

class TM2FixedTriplet(TM2FixedDataset):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        self.k = self.config.data_params.k

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

class TM2FixedCBow(TM2FixedDataset):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        self.k = self.config.data_params.k

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        num_sentences = utterance['total_doc_sentences']
        if sentence_num == 0 or sentence_num == num_sentences-1:
            return None

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        t1 = T-1
        t2 = T+1
        assert t1 < T < t2
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

class TM2FixedTPK(TM2FixedDataset):

    def __init__(
            self,
            train,
            all_dataset,
            config,
            tokenizer_name="GPT2",
            seed=1,
    ):
        """
        """
        super(TM2FixedTPK, self).__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,

        )

    def __getitem__(self, index):
        if self.config.data_params.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
        else:
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.k)
            t = max(0, tp1-self.config.k)

            y_t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence']
            y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]['sentence']
            t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence_id']/utterance['total_doc_sentences']

        y_tm1 = (self.processed_data[index] if (index - 1 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-1]['doc_id']) else self.processed_data[index-1])
        y_tm1 = y_tm1['sentence']
        y_tm2 = (self.processed_data[index] if (index - 2 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-2]['doc_id']) else self.processed_data[index-2])
        y_tm2 = y_tm2['sentence']
        y_tm3 = (self.processed_data[index] if (index - 3 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-3]['doc_id']) else self.processed_data[index-3])
        y_tm3 = y_tm3['sentence']


        result = {
            'y_t': y_t,
            'y_tm1': y_tm1,
            'y_tm2': y_tm2,
            'y_tm3': y_tm3,
            'y_tpk': y_tp1,
        }
        return result
