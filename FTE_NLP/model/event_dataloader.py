from torch.utils.data import DataLoader

def align_labels_with_tokens(labels, word_ids):
    update_labels = []
    for word_id in word_ids:
        label = -100 if word_id is None else labels[word_id]
        update_labels.append(label)
    return update_labels


def tokenize_and_align_labels(dataset,tokenizer):
    tokenized_inputs = tokenizer(
        dataset["text"], truncation=True, padding=True, max_length=256, is_split_into_words=True
    )

    all_labels = dataset["word_tag_id"]
    tag_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        tag_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = tag_labels
    return tokenized_inputs


def train_valid_test(dataset, testsize, validsize):
    """
    split dataset into train test and valid
    :param dataset
    :param testsize
    :param validsize
    :return: dataset
    """
    dataset_all = dataset.train_test_split(test_size=testsize)
    dataset_train_val = dataset_all["train"].train_test_split(test_size=validsize)
    from datasets import DatasetDict
    dataset_split = DatasetDict({
        "train": dataset_train_val["train"],
        "valid": dataset_train_val["test"],
        "test": dataset_all["test"]
    })

    return dataset_split


def dataset_to_dataloader(datasets, data_collator):
    train_dataloader = DataLoader(
        datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    eval_dataloader = DataLoader(
        datasets["valid"], collate_fn=data_collator, batch_size=8
    )

    test_dataloader = DataLoader(
        datasets["test"], collate_fn=data_collator, batch_size=8
    )

    return train_dataloader, eval_dataloader, test_dataloader


#if __name__ == "__main__":
    # load dataset
    # json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
    # event_test = load_dataset("json", data_files=json_filename, split="train")

    # print(event_test)

    # tokenized the text data
    # model_checkpoint = "bert-base-cased"
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # tokenized_datasets = event_test.map(tokenize_and_align_labels,
    #                                     remove_columns=["text", "word_tag", "word_tag_id", "text_tag"], batched=True,)
    #
    # tokenized_datasets = train_valid_test(tokenized_datasets, 0.1, 0.1)
    # print(tokenized_datasets)
    #
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    #
    # train_dataloader, eval_dataloader, test_dataloader = dataset_to_dataloader(tokenized_datasets, data_collator)
