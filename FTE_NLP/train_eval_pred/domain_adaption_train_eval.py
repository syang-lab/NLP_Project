import json
import math
from tqdm.auto import tqdm
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import get_scheduler
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *
from FTE_NLP.utils.early_stop import *

# set device
device = 'cuda' if cuda.is_available() else 'cpu'

# models path
check_point_model_file = '../experiments/models/domain_adaption/distilbert/'

# load file
json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
with open(json_filename) as data_file:
    test_data = json.loads(data_file.read())


# load training, validation and test data
pre_trained_model='distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, use_fast=True)

training_set = event_detection_data(test_data, tokenizer, max_len=512, domain_adaption=True, wwm_prob=0.1)
eval_set = event_detection_data(test_data, tokenizer, max_len=512, domain_adaption=True, wwm_prob=0.1)


# pass data to dataloader
train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0 }
train_loader = DataLoader(training_set, **train_params)

eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0 }
eval_loader = DataLoader(training_set, **train_params)


#2. construct model
model = AutoModelForMaskedLM.from_pretrained(pre_trained_model)

#3. optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#4. learning rate schedular
num_train_epochs = 3


lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_epochs * len(train_loader),
)

#3. train model
progress_bar = tqdm(range(num_train_epochs))
for epoch in range(num_train_epochs):
    train_loss = 0
    nb_train_steps = 0
    nb_train_examples = 0

    # training
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
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

    eval_loss = 0
    nb_eval_steps = 0
    nb_eval_examples = 0

    # evaluation
    model.eval()
    for step, batch in enumerate(eval_loader):
        with torch.no_grad():
            outputs = model(**batch)
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

    progress_bar.update(1)

    # check early stopping
    check_early_stop = earlystop(train_loss, eval_loss)
    if check_early_stop.early_stop:
        model.save_pretrained(check_point_model_file+"early_stop")
        break

    # save check point
    if epoch != 0 and epoch % 100 == 0:
        model.save_pretrained(check_point_model_file+"epoch_"+str(epoch))

model.save_pretrained(check_point_model_file)