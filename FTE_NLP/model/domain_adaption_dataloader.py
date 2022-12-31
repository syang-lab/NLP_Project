import collections
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator


def tokenize_function(data, tokenizer):
    result = tokenizer(data["text"], padding='max_length')
    total_length = len(result["input_ids"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(total_length)]
    print(result["word_ids"][0][:10])
    return result


def chunk_text(data, chunk_size=64):
    """"
    split long single line into chunks, and construct a label
    """
    chunk_data = dict()
    total_length = len(data["input_ids"][0])
    for k in data.keys():
        chunk_data[k] = list()
        for i in range(0, total_length, chunk_size):
            if (i + chunk_size) < total_length:
                chunk_data[k].append(data[k][0][i:i + chunk_size])

    chunk_data["labels"] = chunk_data["input_ids"].copy()
    return chunk_data


def group_texts(examples, chunk_size):
    # concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # we drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(data, tokenizer, wwm_prob):
    for item in data:
        word_ids = item.pop("word_ids")

        # create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
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
        input_ids = item["input_ids"]
        labels = item["labels"]
        new_labels = [-100] * len(labels)

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        item["labels"] = new_labels
    return default_data_collator(data)


def data_preprocess(file, data_cag, tokenizer, chunk_size):
    """"
        preprocess the cleaned dataset
    """
    # load dataset
    data = load_dataset("text", data_files={data_cag: file})

    # tokenized dataset
    data_tokenized = data.map(tokenize_function, batched=True, remove_columns=["text"],
                              fn_kwargs={"tokenizer": tokenizer})

    # chunk text
    data_chunk = data_tokenized.map(chunk_text, batched=True, fn_kwargs={"chunk_size": chunk_size})
    # data_chunk = data_tokenized.map(group_texts, batched=True, fn_kwargs={"chunk_size": chunk_size})

    return data_chunk


if __name__ == "__main__":
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    train_file = "../data/raw_EDT/Domain_adapation/train_cleaned.txt"
    dev_file = "../data/raw_EDT/Domain_adapation/dev_cleaned.txt"

    data_train = data_preprocess(train_file, "train", tokenizer, chunk_size=64)
    data_dev = data_preprocess(dev_file, "dev", tokenizer, chunk_size=64)

    batch_size = 64
    wwm_prob = 0.15

    train_dataloader = DataLoader(
        data_train["train"],
        batch_size=batch_size,
        collate_fn=lambda data: whole_word_masking_data_collator(data, tokenizer, wwm_prob)
    )

    eval_dataloader = DataLoader(
        data_dev["dev"],
        batch_size=batch_size,
        collate_fn=lambda data: whole_word_masking_data_collator(data, tokenizer, wwm_prob)
    )
