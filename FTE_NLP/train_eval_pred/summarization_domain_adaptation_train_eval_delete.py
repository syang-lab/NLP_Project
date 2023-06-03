import math
from tqdm.auto import tqdm
from transformers import T5TokenizerFast
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, random_split
from FTE_NLP.model.summarization_dataset import *
from torch.optim import AdamW
from transformers import get_scheduler


json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'
with open(json_filename) as data_file:
    test_data = json.loads(data_file.read())

model_checkpoint = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint, model_max_length=512)

pre_trained_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, pre_trained_model)

all_dataset = summarization_data(test_data, tokenizer, domain_adaption=True, wwm_prob=0.3)
train_dataset, eval_dataset = random_split(all_dataset, [7, 3], generator=torch.Generator().manual_seed(42))

data_collator = DataCollatorForSeq2Seq(tokenizer, pre_trained_model)

# pass data to dataloader
train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
train_loader = DataLoader(train_dataset, collate_fn=data_collator, **train_params)

eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, **train_params)

# define optimizer
optimizer = AdamW(pre_trained_model.parameters(), lr=2e-5)


# define learning rate scheduler
num_train_epochs = 10
num_update_steps_per_epoch = len(train_loader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# training and evaluate
progress_bar = tqdm(range(num_train_epochs))
for epoch in range(num_train_epochs):
    train_loss = 0
    nb_train_steps = 0
    nb_train_examples = 0
    # training
    pre_trained_model.train()
    for batch in train_loader:

        optimizer.zero_grad()

        outputs = pre_trained_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        nb_train_steps += 1
        nb_train_examples += len(batch)
        lr_scheduler.step()

    train_epoch_loss = train_loss / nb_train_steps
    print(f"Epoch {epoch}, Training Loss: {train_epoch_loss}")

    try:
        perplexity = math.exp(train_epoch_loss)
    except OverflowError:
        perplexity = float("inf")
    print(f">>> Epoch {epoch}: Training Perplexity: {perplexity}")
    progress_bar.update(1)

    eval_loss = 0
    nb_eval_steps = 0
    nb_eval_examples = 0

    pre_trained_model.eval()
    for batch in train_loader:
        with torch.no_grad():
            outputs = pre_trained_model(**batch)
        loss = outputs.loss
        eval_loss += loss.item()

        nb_eval_steps += 1
        nb_eval_examples += len(batch)

    eval_epoch_loss = eval_loss / nb_eval_steps
    print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss}")

    try:
        eval_perplexity = math.exp(eval_epoch_loss)
    except OverflowError:
        eval_perplexity = float("inf")
    print(f">>> Epoch {epoch}: Evaluation Perplexity: {eval_perplexity}")


# save model
# model_save_filename = "../experiments/model_bucket/mt5-small-pretrained"
# model.save_pretrained(model_save_filename)