import json
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from FTE_NLP.model.event_dataloader import *
from FTE_NLP.utils.post_process import postprocess_event
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate


# load dataset
json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
event_test = load_dataset("json", data_files=json_filename, split="train")
# print(event_test)

# model
model_checkpoint = "bert-base-cased"

# construct tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# tokenized dataset
tokenized_event = event_test.map(tokenize_and_align_labels,
                                 remove_columns=["text", "word_tag", "word_tag_id", "text_tag", "text_tag_id"],
                                 batched=True,
                                 fn_kwargs={"tokenizer": tokenizer}
                                 )

# train and test split
tokenized_event = train_valid_test(tokenized_event, 0.05, 0.1)
# print(tokenized_event)

# construct dataloader
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
train_dataloader, eval_dataloader, test_dataloader = dataset_to_dataloader(tokenized_event, data_collator)

# construct model
tags2ids_name = "../data/raw_EDT/Event_detection/tags2ids.json"
ids2tags_name = "../data/raw_EDT/Event_detection/ids2tags.json"

with open(tags2ids_name) as json_file:
    tags2ids = json.load(json_file)

with open(ids2tags_name) as json_file:
    ids2tags = json.load(json_file)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=ids2tags,
    label2id=tags2ids,
)

# define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# define schedular
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch


lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# define evaluation matrix
metric = evaluate.load("seqeval")

# train and evaluation
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    # training
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        progress_bar.update(1)

    # eval
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        preds_tag, labels_tag = postprocess_event(predictions, labels, ids2tags)
        metric.add_batch(predictions=preds_tag, references=labels_tag)

    # TODO: define early stop
    results = metric.compute()

    print(
       f"epoch {epoch}:",
       {
           key: results[f"overall_{key}"]
           for key in ["precision", "recall", "f1", "accuracy"]
       },
    )

# save model
model_save_filename = "../experiments/models/bert-base-cased-pretrained"
model.save_pretrained(model_save_filename)

