import sys
import os
import random
import pickle
import tqdm
import torch
import numpy as np
from encoder.src import constants

from encoder.src.datasets import encoder

class GenomeSARS(encoder.BaseDataset):

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
        self.section_names = []
        self.section_ids = []

        self.data_dir = constants.PATH2GENOMESARS
        if type(self.train) is str:
            print ('VAL DATASET')
            self.fname = os.path.join(self.data_dir, "valid.pkl")
        elif self.train:
            print ('TRAIN DATASET')
            self.fname = os.path.join(self.data_dir, "train.pkl")
        else:
            print ('TEST DATASET')
            self.fname = os.path.join(self.data_dir, "test.pkl")

    def _process_data(self):
        self.processed_data = []
        doc_id = 0
        min_length = np.inf
        #split_pattern = ". "
        split_pattern = " [SEP] "
        data = pickle.load(open(self.fname, "rb"))
        random.seed(1234)
        print ('SHUFFLING STORIES')
        random.shuffle(data)
        for example in tqdm.tqdm(data):
            #import pdb; pdb.set_trace()
            story = example[0]+ ' '
            story = story.replace('. ', ' [SEP] ')
            title, text = story.split('\n')

            text = text.split(split_pattern)
            text = text[:-1]

            for i in range(len(text)):
                # if i==0:
                #     text[i] = f'<|endoftext|> [SEP] {text[i]} [SEP] '
                if i==(len(text)-1):
                    text[i] = f'{text[i]} [SEP] <|endoftext|>'
                else:
                    text[i] = f'{text[i]} [SEP] '

            # text[0] = f'<|endoftext|> [SEP] {text[0]} [SEP]'
            # text[-1] = f'{text[-1]} [SEP] <|endoftext|>'
            story = text
            # print(text[0])
            # print(text[1])
            # print(text[-2])
            # print(text[-1])
            # exit()

            # title = 'Unknown Title'
            # title = f'<|endoftext|> {title}'
            # text = f'{text} <|endoftext|>'
            # text = text.split(split_pattern)
            # #text = [t + split_pattern for t in text if t[-1] != "."]
            # story = [title] + text
            # story = [t + split_pattern for t in story[:-1]] + story[-1:]

            if len(story) <= 3:
                continue
            for sentence_counter, sentence in enumerate(story):
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_counter,
                    "doc_id": doc_id,
                    "total_doc_sentences": len(story)
                }
                self.processed_data.append(sentence_info)
            doc_id += 1

    def __len__(self):
        return len(self.processed_data)


class GenomeSARSCBow(GenomeSARS):

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
        #import pdb; pdb.set_trace()

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
