import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict



class event_detection_data(Dataset):
    def __init__(self, raw_data, tokenizer, max_len, domain_adaption=False, wwm_prob=0.1):
        self.len = len(raw_data)
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.domain_adaption = domain_adaption
        self.wwm_prob = wwm_prob

    def __getitem__(self, index):
        tokenized_inputs = self.tokenizer(
            self.data[index]["text"],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            is_split_into_words=True
        )

        ids = tokenized_inputs['input_ids']
        mask = tokenized_inputs['attention_mask']

        if self.domain_adaption:
            if self.tokenizer.is_fast:
                input_ids, labels = self._whole_word_masking(self.tokenizer, tokenized_inputs, self.wwm_prob)
                return {
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(mask),
                    'labels': torch.tensor(labels, dtype=torch.long)
                }
            else:
                print("requires fast tokenizer for word_ids")
        else:
            return {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor(mask),
                'targets': torch.tensor(self.data[index]["text_tag_id"][0], dtype=torch.long)
            }

    def __len__(self):
        return self.len

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