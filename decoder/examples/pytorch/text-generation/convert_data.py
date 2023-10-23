from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


split = 'test'
split = 'val'
split = 'train'
file_path = f'../../../../../data/wikisection/wikisection_withSections.{split}.txt'
output_path = f'{split}.txt'
#import pdb; pdb.set_trace()

with open(file_path, encoding="utf-8") as f:
    with open(output_path, 'w') as fout:
        for idx, row in enumerate(f.readlines()):
            if row.strip():
                # Text for GPT2
                row = row.strip() # NOTE: remove break line
                row = row.replace(". ", " . ") # NOTE marking end of sentence
                row = f"{tokenizer.bos_token} {row} {tokenizer.eos_token}"
                tokenized_text = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(row))
                if len(tokenized_text) >= 1024:
                    continue
                fout.write(row + '\n')

