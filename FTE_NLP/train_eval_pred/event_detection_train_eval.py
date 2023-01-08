import json
import torch
from tqdm.auto import tqdm
from torch import cuda
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers import get_scheduler
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *
from FTE_NLP.utils.early_stop import *

# set device
device = 'cuda' if cuda.is_available() else 'cpu'

# models path
save_check_point_model = '../experiments/models/event_detection/distilbert/'

# load file
json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
with open(json_filename) as data_file:
    test_data = json.loads(data_file.read())

# load training, validation and test data
pre_trained_model = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=True)
all_dataset = event_detection_data(test_data, tokenizer, max_len=512)

train_dataset, eval_dataset = random_split(all_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

# pass data to dataloader
train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
train_loader = DataLoader(train_dataset, **train_params)

eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
eval_loader = DataLoader(eval_dataset, **train_params)

# initialize model
checkpoint_model = '../experiments/models/domain_adaption/distilbert/'
model = DistillBERTClass(checkpoint_model)

# define loss function
loss_function = torch.nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

# define schedular
num_train_epochs = 3

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_epochs * len(train_loader),
)

progress_bar = tqdm(range(num_train_epochs))
for epoch in range(num_train_epochs):
    train_loss = 0
    nb_train_correct = 0
    nb_train_steps = 0
    nb_train_examples = 0

    model.train()
    for _, data in enumerate(train_loader):
        ids = data['input_ids']
        mask = data['attention_mask']
        targets = data['targets']

        outputs = model(ids, mask)

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        max_val, max_idx = torch.max(outputs.data, dim=1)
        nb_train_correct += (max_idx == targets).sum().item()

        nb_train_steps += 1
        nb_train_examples += targets.size(0)

        # update learning rate and scale bar
        lr_scheduler.step()
        progress_bar.update(1)

    train_epoch_loss = train_loss / nb_train_steps
    train_epoch_accu = (nb_train_correct * 100) / nb_train_examples
    print(f"Epoch {epoch}, Training Loss: {train_epoch_loss}")
    print(f"Epoch {epoch}, Training Accuracy: {train_epoch_accu}")

    eval_loss = 0
    nb_eval_steps = 0
    nb_eval_correct = 0
    nb_eval_examples = 0

    model.eval()
    for _, data in enumerate(eval_loader):
        ids = data['input_ids']
        mask = data['attention_mask']
        targets = data['targets']

        with torch.no_grad():
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)

        eval_loss += loss.item()

        max_val, max_idx = torch.max(outputs.data, dim=1)
        nb_eval_correct += (max_idx == targets).sum().item()

        nb_eval_steps += 1
        nb_eval_examples += targets.size(0)

    eval_epoch_loss = eval_loss / nb_eval_steps
    eval_epoch_accu = (nb_eval_correct * 100) / nb_eval_examples
    print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss}")
    print(f"Epoch {epoch}, Evaluation Accuracy: {eval_epoch_accu}")

    progress_bar.update(1)

    # check early stopping
    check_early_stop = earlystop(train_loss, eval_loss)
    if check_early_stop.early_stop:
        torch.save(model, save_check_point_model + "early_stop")
        break

    # save check point
    if epoch != 0 and epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate_schedular': lr_scheduler.state_dict()
        }, save_check_point_model + "epoch_" + str(epoch))
