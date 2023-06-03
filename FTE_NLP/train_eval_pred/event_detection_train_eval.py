import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_scheduler
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *
from FTE_NLP.utils.early_stop import *

# TODO:prediction
# # model_bucket path
# save_check_point_model = '../experiments/model_bucket/event_detection/distilbert/'
#
# # load file
# json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
# with open(json_filename) as data_file:
#     test_data = json.loads(data_file.read())
#
# # load training, validation and test data
# pre_trained_model = 'distilbert-base-cased'
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=True)
# all_dataset = event_detection_data(test_data, tokenizer, max_len=512)
#
# print(f"number of entries in the dataset {len(all_dataset)}")
#
# train_size = int(all_dataset[0] * len(all_dataset))
# test_size = len(all_dataset) - train_size
#
# train_dataset, eval_dataset = random_split(all_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
#
# # pass data to dataloader
# train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# train_loader = DataLoader(all_dataset, **train_params)
# eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0}
# eval_loader = DataLoader(all_dataset, **train_params)
#
# # initialize model
# checkpoint_model = '../experiments/model_bucket/domain_adaption/distilbert/'
# model = DistillBERTClass(checkpoint_model)
#
# # define loss function
# loss_function = torch.nn.CrossEntropyLoss()
#
# # define optimizer
# optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
#
# # define schedular
# num_train_epochs = 3
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_train_epochs * len(train_loader))
#
#
# progress_bar = tqdm(range(num_train_epochs))
# for epoch in range(num_train_epochs):
#     train_loss = 0
#     nb_train_correct = 0
#     nb_train_steps = 0
#     nb_train_examples = 0
#
#     model.train()
#     for _, data in enumerate(train_loader):
#         ids = data['input_ids']
#         mask = data['attention_mask']
#         targets = data['targets']
#
#         outputs = model(ids, mask)
#
#         optimizer.zero_grad()
#         loss = loss_function(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#         max_val, max_idx = torch.max(outputs.data, dim=1)
#         nb_train_correct += (max_idx == targets).sum().item()
#
#         nb_train_steps += 1
#         nb_train_examples += targets.size(0)
#
#         # update learning rate and scale bar
#         lr_scheduler.step()
#         progress_bar.update(1)
#
#     train_epoch_loss = train_loss / nb_train_steps
#     train_epoch_accu = (nb_train_correct * 100) / nb_train_examples
#     print(f"Epoch {epoch}, Training Loss: {train_epoch_loss}")
#     print(f"Epoch {epoch}, Training Accuracy: {train_epoch_accu}")
#
#     eval_loss = 0
#     nb_eval_steps = 0
#     nb_eval_correct = 0
#     nb_eval_examples = 0
#
#     model.eval()
#     for _, data in enumerate(eval_loader):
#         ids = data['input_ids']
#         mask = data['attention_mask']
#         targets = data['targets']
#
#         with torch.no_grad():
#             outputs = model(ids, mask)
#             loss = loss_function(outputs, targets)
#
#         eval_loss += loss.item()
#
#         max_val, max_idx = torch.max(outputs.data, dim=1)
#         nb_eval_correct += (max_idx == targets).sum().item()
#
#         nb_eval_steps += 1
#         nb_eval_examples += targets.size(0)
#
#     eval_epoch_loss = eval_loss / nb_eval_steps
#     eval_epoch_accu = (nb_eval_correct * 100) / nb_eval_examples
#     print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss}")
#     print(f"Epoch {epoch}, Evaluation Accuracy: {eval_epoch_accu}")
#
#     progress_bar.update(1)

    # # check early stopping
    # check_early_stop = earlystop(train_loss, eval_loss)
    # if check_early_stop.early_stop:
    #     torch.save(model, save_check_point_model + "early_stop")
    #     break
    #
    # # save check point
    # if epoch != 0 and epoch % 100 == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'learning_rate_schedular': lr_scheduler.state_dict()
    #     }, save_check_point_model + "epoch_" + str(epoch))

def event_detection_train_eval(train_eval_filename,
                               token_pre_trained_model,
                               train_eval_split,
                               token_max_len,
                               checkpoint_model,
                               num_train_epochs,
                               train_batch_size,
                               eval_batch_size,
                               train_num_workers,
                               eval_num_workers,
                               learning_rate,
                               save_check_point_model):

    # load file
    with open(train_eval_filename) as data_file:
        train_eval_data = json.loads(data_file.read())

    # load training, validation and test data
    tokenizer = AutoTokenizer.from_pretrained(token_pre_trained_model, use_fast=True)
    train_eval_dataset = event_detection_data(train_eval_data, tokenizer, max_len=token_max_len)

    print(f"number of entries in the dataset {len(train_eval_dataset)}")

    train_size = int(train_eval_split[0] * len(train_eval_dataset))
    test_size = len(train_eval_dataset) - train_size

    train_dataset, eval_dataset = random_split(train_eval_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

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
                'learning_rate_schedular': lr_scheduler.state_dict()},
                save_check_point_model + "event_detection"+"epoch_" + str(epoch))

    return



if __name__ == "__main__":
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # input path
        train_eval_filename = '../data/raw_EDT/Event_detection/dev.json'
        train_eval_split = [0.5, 0.5]
        token_max_len = 512

        # pretrained model
        token_pre_trained_model = 'distilbert-base-cased'

        # tokenizer
        max_len = 512
        domain_adaption = False
        mask_prob = 0

        # adapted model
        #checkpoint_model='distilbert-base-cased'
        checkpoint_model = '../experiments/model_bucket/domain_adaption/distilbert/event_domain_adaption'

        # training and evaluation
        num_train_epochs = 15
        train_batch_size = 16
        eval_batch_size = 16
        train_num_workers = 1
        eval_num_workers = 1
        learning_rate = 1e-4

        save_check_point_model = '../experiments/model_bucket/event_detection/distilbert/'

        event_detection_train_eval(train_eval_filename,
                                   token_pre_trained_model,
                                   train_eval_split,
                                   token_max_len,
                                   checkpoint_model,
                                   num_train_epochs,
                                   train_batch_size,
                                   eval_batch_size,
                                   train_num_workers,
                                   eval_num_workers,
                                   learning_rate,
                                   save_check_point_model)
