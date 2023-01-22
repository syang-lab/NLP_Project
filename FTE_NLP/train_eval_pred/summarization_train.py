from tqdm.auto import tqdm
import torch
import numpy as np
import evaluate
from FTE_NLP.model.summarizaton_dataloader import *
from FTE_NLP.utils.post_process import *

# load dataset
data_file_name = "../data/raw_EDT/Trading_benchmark/evaluate_news.json"
_, data_small = json2dataset(data_file_name)
# clean dataset
column_names = ['pub_time', 'labels']
data_small = drop_column(data_small, column_names)
data_small = data_small.map(summarize_data_clean, batched=True)

# train valid and test split
data_small = train_valid_test(data_small, testsize=0.2, validsize=0.1)

# define model and tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# tokenized dataset
data_small = data_small.map(dataset_tokenized, batched=True,
                            fn_kwargs={"tokenizer": tokenizer, "max_input_length": 512, "max_label_length": 30})
data_small = drop_column(data_small, ["text", "title"])

# construct dataloader
data_collator = DataCollatorForSeq2Seq(tokenizer, model)
train_dataloader, test_dataloader, valid_dataloader = dataset_dataloader(data_small, data_collator, model, tokenizer, batch_size=8)

# define optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

# define learning rate scheduler
from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# define evaluation matrix
rouge_score = evaluate.load("rouge")


# training and evaluate
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

    model.eval()
    for batch in train_dataloader:
        with torch.no_grad():
            generate_token = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"])
            decoded_generate = tokenizer.batch_decode(generate_token, skip_special_tokens=True)

            labels = batch["labels"]
            labels = labels.numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_generate, decoded_labels = postprocess_text(
                decoded_generate, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_generate, references=decoded_labels)

    result = rouge_score.compute()
    # Extract the median ROUGE scores
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)


# save model
model_save_filename = "../experiments/model_bucket/mt5-small-pretrained"
model.save_pretrained(model_save_filename)