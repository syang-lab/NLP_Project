import torch
import math
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from FTE_NLP.model.domain_adaption_dataloader import whole_word_masking_data_collator
from FTE_NLP.model.domain_adaption_dataloader import data_preprocess
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm


#1. construct dataset
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

train_file = "../data/raw_EDT/Domain_adapation/train_test_cleaned.txt"
dev_file = "../data/raw_EDT/Domain_adapation/dev_test_cleaned.txt"

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


#2. construct model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

#3. optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

#4. learning rate schedular
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#3. train model
progress_bar = tqdm(range(num_train_epochs))
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

    # evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(batch_size))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataloader)]

    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")


#model_save_filename = "../experiments/models/distilbert-base-uncased-pretrained"
#model.save_pretrained(model_save_filename)
