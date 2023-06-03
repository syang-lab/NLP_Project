import json
import re
import torch
import numpy as np
from collections import defaultdict

class summarization_data:
    def __init__(self, raw_data, tokenizer, domain_adaption=False, wwm_prob=0.1):
        data = self.select_data(raw_data)
        self.data = self.data_clean(data)
        self.len = len(self.data)
        self.tokenizer = tokenizer
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
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        text_ids = tokenized_text['input_ids']
        text_mask = tokenized_text['attention_mask']

        tokenized_title = self.tokenizer(
            self.data[index]["title"],
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        title_ids = tokenized_title['input_ids']

        if self.domain_adaption:
            if self.tokenizer.is_fast:
                input_ids, labels = self._word_masking(self.tokenizer, tokenized_text, self.wwm_prob)
                return {
                    'input_ids': torch.tensor(input_ids),
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

    def _word_masking(self, tokenizer, tokenized_inputs, wwm_prob):
        # randomly mask words
        input_ids = tokenized_inputs["input_ids"]
        mask = np.random.binomial(1, wwm_prob, (len(input_ids),))

        # labels only contains masked words as target
        labels = [-100] * len(input_ids)

        for idx in np.where(mask == 1)[0]:
            labels[idx] = input_ids[idx]
            #add special sentinel tokens
            sentinel_token = tokenizer.additional_special_tokens[input_ids[idx] % 100]
            input_ids[idx] = tokenizer(sentinel_token).input_ids[0]
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
# from transformers import T5Tokenizer, T5TokenizerFast
#
# json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'
# with open(json_filename) as data_file:
#     test_data = json.loads(data_file.read())
#
#
# model_checkpoint = "t5-small"
# tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint, model_max_length=512)
#
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
# all_dataset = summarization_domain_adaptation_data(test_data, tokenizer, domain_adaption=True, wwm_prob=0.1)
#
# train_dataset, eval_dataset = random_split(all_dataset, [7, 3], generator=torch.Generator().manual_seed(42))
#
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
# # pass data to dataloader
#
# train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# train_loader = DataLoader(train_dataset, collate_fn=data_collator, **train_params)
#
# eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, **train_params)
#
#
# for item in train_loader:
#     print(item)
#     break