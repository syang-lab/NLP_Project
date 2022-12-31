from datasets import load_dataset
import re
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def json2dataset(datafile_name):
    """
    load json file and return dataset
    :param datafile_name
    :return: datadict
    """
    data_all = load_dataset("json", data_files=datafile_name, split="train")
    data_small = data_all.select([0, 1, 2, 3, 4, 5])
    return data_all, data_small


def drop_column(dataset, column_names):
    """
    remove columns
    :param dataset
    :param column_names
    :return: dataset
    """
    dataset = dataset.remove_columns(column_names)
    return dataset


def summarize_data_clean(dataset):
    """
    clean the data in the text and title
    :param dataset
    :return: dataset
    """
    for key in dataset:
        for item in range(len(dataset[key])):
            dataset[key][item] = re.sub(r"[?|!+?|:|(|)]|\\|-|/.*?/|http\S+", "", dataset[key][item].lower())
    return dataset


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


def dataset_tokenized(dataset, tokenizer, max_input_length, max_label_length):
    """
    tokenized the text and title in the dataset
    :param dataset
    :param tokenizer
    :param max_input_length
    :param max_label_length
    :return: dataset
    """
    dataset_tokenized = tokenizer(
        dataset["text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        dataset["title"],
        max_length=max_label_length,
        padding="max_length",
        truncation=True,
    )
    dataset_tokenized["labels"] = labels["input_ids"]
    return dataset_tokenized


def dataset_dataloader(dataset, data_collator, model, tokenizer, batch_size=8):
    """
    construct dataloader
    :param dataset:
    :param data_collector:
    :param model:
    :param tokenizer:
    :param batch_size:
    :return: dataloader
    """
    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    test_dataloader = DataLoader(
        dataset["test"], collate_fn=data_collator, batch_size=batch_size
    )

    eval_dataloader = DataLoader(
        dataset["valid"], collate_fn=data_collator, batch_size=batch_size
    )
    return train_dataloader, test_dataloader, eval_dataloader


if __name__ == "main":
    data_file_name = "../data/raw_EDT/Trading_benchmark/evaluate_news.json"
    _, data_small = json2dataset(data_file_name)
    # print(data_small)

    column_names = ['pub_time', 'labels']
    data_small = drop_column(data_small, column_names)

    data_small = data_small.map(summarize_data_clean, batched=True)

    data_small = train_valid_test(data_small, testsize=0.2, validsize=0.1)

    # print(data_small)

    model_checkpoint = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)

    data_small = data_small.map(dataset_tokenized, batched=True,
                                fn_kwargs={"tokenizer": tokenizer, "max_input_length": 512, "max_label_length": 30})

    data_small = drop_column(data_small, ["text", "title"])

    #print(data_small)