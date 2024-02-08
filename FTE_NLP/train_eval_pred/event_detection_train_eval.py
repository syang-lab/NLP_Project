import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers import get_scheduler
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *
from FTE_NLP.utils.early_stop import *

def event_detection_train_eval(train_eval_filename,
                               pre_trained_model,
                               train_eval_split,
                               token_max_len,
                               train_batch_size,
                               eval_batch_size,
                               train_num_workers,
                               eval_num_workers,
                               checkpoint_model,
                               learning_rate,
                               num_train_epochs,
                               save_check_point_model
                               ):
    # load file
    with open(train_eval_filename) as data_file:
        train_eval_data = json.loads(data_file.read())

    # load training, validation and test data
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, use_fast=True)
    train_eval_dataset = event_detection_data(train_eval_data, tokenizer, max_len=token_max_len)

    train_dataset, eval_dataset = random_split(train_eval_dataset, train_eval_split,
                                               generator=torch.Generator().manual_seed(42))

    # pass data to dataloader
    train_params = {'batch_size': train_batch_size, 'shuffle': True, 'num_workers': train_num_workers}
    train_loader = DataLoader(train_dataset, **train_params)

    eval_params = {'batch_size': eval_batch_size, 'shuffle': True, 'num_workers': eval_num_workers}
    eval_loader = DataLoader(eval_dataset, **eval_params)

    # initialize model
    model = DistillBERTClass(checkpoint_model)

    # define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # define schedular
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

    return
