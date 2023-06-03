import json
import math
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_scheduler
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *
from FTE_NLP.utils.early_stop import *

# # model_bucket path
# save_check_point_model = '../experiments/model_bucket/domain_adaption/distilbert/'
#
# # load file
# json_filename = '../data/raw_EDT/Event_detection/dev_test.json'
# with open(json_filename) as data_file:
#     test_data = json.loads(data_file.read())
#
# # load training, validation and test data
# pre_trained_model='distilbert-base-cased'
# tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, use_fast=True)
#
#
# training_set = event_detection_data(test_data, tokenizer, max_len=512, domain_adaption=True, wwm_prob=0.1)
# eval_set = event_detection_data(test_data, tokenizer, max_len=512, domain_adaption=True, wwm_prob=0.1)
#
#
# # pass data to dataloader
# train_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0 }
# train_loader = DataLoader(training_set, **train_params)
#
# eval_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 0 }
# eval_loader = DataLoader(training_set, **train_params)
#
#
# #2. construct model
# model = AutoModelForMaskedLM.from_pretrained(pre_trained_model)
#
# #3. optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#
# #4. learning rate schedular
# num_train_epochs = 3
#
#
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_train_epochs * len(train_loader),
# )
#
# #3. train model
# progress_bar = tqdm(range(num_train_epochs))
# for epoch in range(num_train_epochs):
#     train_loss = 0
#     nb_train_steps = 0
#     nb_train_examples = 0
#
#     # training
#     model.train()
#     for batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#         nb_train_steps += 1
#         nb_train_examples += len(batch)
#
#         lr_scheduler.step()
#
#     train_epoch_loss = train_loss / nb_train_steps
#     print(f"Epoch {epoch}, Training Loss: {train_epoch_loss}")
#
#     try:
#         perplexity = math.exp(train_epoch_loss)
#     except OverflowError:
#         perplexity = float("inf")
#     print(f">>> Epoch {epoch}: Training Perplexity: {perplexity}")
#
#     eval_loss = 0
#     nb_eval_steps = 0
#     nb_eval_examples = 0
#
#     # evaluation
#     model.eval()
#     for step, batch in enumerate(eval_loader):
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#
#         eval_loss += loss.item()
#
#         nb_eval_steps += 1
#         nb_eval_examples += len(batch)
#
#     eval_epoch_loss = eval_loss / nb_eval_steps
#     print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss}")
#
#     try:
#         eval_perplexity = math.exp(eval_epoch_loss)
#     except OverflowError:
#         eval_perplexity = float("inf")
#     print(f">>> Epoch {epoch}: Evaluation Perplexity: {eval_perplexity}")
#
#     progress_bar.update(1)

# # check early stopping
# check_early_stop = earlystop(train_loss, eval_loss)
# if check_early_stop.early_stop:
#     model.save_pretrained(save_check_point_model+"early_stop")
#     break

# # save check point
# if epoch != 0 and epoch % 100 == 0:
#     model.save_pretrained(save_check_point_model+"epoch_"+str(epoch))

# # prediction
# output_print = batch['labels'].numpy()
# output_print = np.where(output_print != -100, output_print, tokenizer.pad_token_id)
# output_print = tokenizer.batch_decode(output_print, skip_special_tokens=True)
# print(f"output print : {output_print}")

# # print predicted token
# mask_token_index = torch.where(batch["input_ids"] == tokenizer.mask_token_id)[1]
# logits = outputs.logits
# mask_token_logits = logits[:, mask_token_index, :]
# output_token = torch.topk(mask_token_logits, 1, dim=2).indices[0]
# print(f"prediction output print: {tokenizer.batch_decode(output_token)}")

def domain_adaption_train_eval(train_eval_filename,
                               token_pre_trained_model,
                               train_eval_split,
                               token_max_len,
                               domain_adaption,
                               mask_prob,
                               num_train_epochs,
                               train_batch_size,
                               eval_batch_size,
                               learning_rate,
                               train_num_workers,
                               eval_num_workers,
                               save_check_point_model):

    # load file
    with open(train_eval_filename) as data_file:
        train_eval_data = json.loads(data_file.read())

    # load training, validation and test data
    tokenizer = AutoTokenizer.from_pretrained(token_pre_trained_model, use_fast=True)
    training_eval_dataset = event_detection_data(train_eval_data, tokenizer, max_len=token_max_len,
                                                 domain_adaption=domain_adaption, wwm_prob=mask_prob)

    print(f"number of entries in the dataset {len(training_eval_dataset)}")

    train_size = int(train_eval_split[0] * len(training_eval_dataset))
    test_size = len(training_eval_dataset) - train_size

    train_dataset, eval_dataset = random_split(training_eval_dataset, [train_size,test_size],
                                               generator=torch.Generator().manual_seed(42))

    # pass data to dataloader
    train_params = {'batch_size': train_batch_size, 'shuffle': True, 'num_workers': train_num_workers}
    train_loader = DataLoader(train_dataset, **train_params)

    eval_params = {'batch_size': eval_batch_size, 'shuffle': True, 'num_workers': eval_num_workers}
    eval_loader = DataLoader(eval_dataset, **eval_params)

    # 2. construct model
    model = AutoModelForMaskedLM.from_pretrained(token_pre_trained_model)

    # 3. optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. learning rate schedular
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_epochs * len(train_loader),
    )

    # 3. train model
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
        print(f"Epoch {epoch}, Training Loss: {train_epoch_loss:.2f}")

        try:
            perplexity = math.exp(train_epoch_loss)
        except OverflowError:
            perplexity = float("inf")
        print(f">>> Epoch {epoch}: Training Perplexity: {perplexity:.2f}")

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
        print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss:.2f}")

        try:
            eval_perplexity = math.exp(eval_epoch_loss)
        except OverflowError:
            eval_perplexity = float("inf")
        print(f">>> Epoch {epoch}: Evaluation Perplexity: {eval_perplexity:.2f}")

        progress_bar.update(1)

        # check early stopping
        check_early_stop = earlystop(train_loss, eval_loss)
        if check_early_stop.early_stop:
            model.save_pretrained(save_check_point_model + "early_stop")
            break

        # save check point
        if epoch != 0 and epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate_schedular': lr_scheduler.state_dict()
            }, save_check_point_model + "event_domain_adaption"+"epoch_" + str(epoch))

            model.save_pretrained(save_check_point_model + "event_domain_adaption")
    return

if __name__=="__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #input path
    train_eval_filename = '../data/raw_EDT/Event_detection/dev.json'
    train_eval_split = [0.7,0.3]
    token_max_len = 512

    #pretrained model
    token_pre_trained_model = 'distilbert-base-cased'

    #tokenizer
    max_len = 512
    domain_adaption = True
    mask_prob = 0.1

    #training and evaluation
    num_train_epochs = 5
    train_batch_size = 16
    eval_batch_size = 16
    train_num_workers = 1
    eval_num_workers = 1
    learning_rate = 1e-4

    save_check_point_model = '../experiments/model_bucket/domain_adaption/distilbert/'

    domain_adaption_train_eval(train_eval_filename,
                               token_pre_trained_model,
                               train_eval_split,
                               token_max_len,
                               domain_adaption,
                               mask_prob,
                               num_train_epochs,
                               train_batch_size,
                               eval_batch_size,
                               learning_rate,
                               train_num_workers,
                               eval_num_workers,
                               save_check_point_model)