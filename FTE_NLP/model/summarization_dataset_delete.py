import json
import re
import torch
import numpy as np
from collections import defaultdict

class summarization_data:
    def __init__(self, raw_data, tokenizer, input_max_length, domain_adaption=False, wwm_prob=0.1):
        data = self.select_data(raw_data)
        self.data = self.data_clean(data)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.padding_max_len = "max_length"
        self.domain_adaption = domain_adaption
        self.wwm_prob = wwm_prob

    def select_data(self, raw_data):
        data = list()
        for item in raw_data:
            del item['pub_time']
            del item['labels']
            data.append(item)
        return data


    def data_clean(self, data):
        for item in data:
            item["text"] = re.sub(r"[?|!+?|:|(|)]|\\|-|/.*?/|http\S+", "", item["text"].lower())
            item["title"] = re.sub(r"[?|!+?|:|(|)]|\\|-|/.*?/|http\S+", "", item["title"].lower())
        return data

    def __getitem__(self, index):
        tokenized_text = self.tokenizer(
            self.data[index]["text"],
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            is_split_into_words=True
        )
        text_ids = tokenized_text['input_ids']
        text_mask = tokenized_text['attention_mask']

        tokenized_title = self.tokenizer(
            self.data[index]["title"],
            max_length=self.input_max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            is_split_into_words=True
        )

        title_ids = tokenized_title['input_ids']

        if self.domain_adaption:
            if self.tokenizer.is_fast:
                _, labels = self._whole_word_masking(self.tokenizer, tokenized_text, self.wwm_prob)
                return {
                    'input_ids': torch.tensor(text_ids),
                    'attention_mask': torch.tensor(text_mask),
                    'labels': torch.tensor(labels)
                }
            else:
                print("requires fast tokenizer for word_ids")
        else:
            return {
                'input_ids': torch.tensor(text_ids),
                'attention_mask': torch.tensor(text_mask),
                'labels': torch.tensor(title_ids)
            }

    def _whole_word_masking(self, tokenizer, tokenized_inputs, wwm_prob):
        word_ids = tokenized_inputs.word_ids(0)

        # create a map between words_ids and natural id
        mapping = defaultdict(list)
        current_word_index = -1
        current_word = None

        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # randomly mask words
        mask = np.random.binomial(1, wwm_prob, (len(mapping),))
        input_ids = tokenized_inputs["input_ids"]

        # labels only contains masked words as target
        labels = [-100] * len(input_ids)

        for word_id in np.where(mask == 1)[0]:
            for idx in mapping[word_id]:
                labels[idx] = tokenized_inputs["input_ids"][idx]
                input_ids[idx] = tokenizer.mask_token_id
        return input_ids, labels


    def __len__(self):
        return self.len



# from tqdm.auto import tqdm
# from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM
# from transformers import DataCollatorForSeq2Seq
# from torch.utils.data import DataLoader, random_split
# from FTE_NLP.model.summarization_dataset import *
# from torch.optim import AdamW
# from transformers import get_scheduler
# from FTE_NLP.utils.post_process import *
#
# json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'
# with open(json_filename) as data_file:
#     test_data = json.loads(data_file.read())
#
# model_checkpoint = "t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
# all_dataset = summarization_data(test_data, tokenizer, input_max_length=512, domain_adaption=False, wwm_prob=0.1)
# train_dataset, eval_dataset = random_split(all_dataset, [7, 3], generator=torch.Generator().manual_seed(42))
#
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
#
# print(train_dataset[0])
#
# # pass data to dataloader
# train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# train_loader = DataLoader(train_dataset, collate_fn=data_collator, **train_params)
#
# eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, **train_params)