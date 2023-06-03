from tqdm.auto import tqdm
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, random_split
from FTE_NLP.model.summarization_dataset import *
from torch.optim import AdamW
from transformers import get_scheduler
from FTE_NLP.utils.post_process import *
from FTE_NLP.utils.early_stop import *

import numpy as np
import evaluate

# save_check_point_model = '../experiments/model_bucket/domain_adaption/t5/'
# json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'
# with open(json_filename) as data_file:
#     test_data = json.loads(data_file.read())
#
#
# model_checkpoint = "t5-small"
# token_max_len=512
# tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint, model_max_length=token_max_len)
#
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
# domain_adaption=True
# wwm_prob=0.1
# training_eval_dataset = summarization_data(test_data, tokenizer, domain_adaption=domain_adaption, wwm_prob=wwm_prob)
# print(f"number of entries in the dataset {len(training_eval_dataset)}")
#
# train_eval_split=[0.7,0.3]
# train_size = int(train_eval_split[0] * len(training_eval_dataset))
# test_size = len(training_eval_dataset) - train_size
#
# train_dataset, eval_dataset = random_split(training_eval_dataset, [7, 3], generator=torch.Generator().manual_seed(42))
#
# data_collator = DataCollatorForSeq2Seq(tokenizer, model)
#
#
# # pass data to dataloader
# train_batch_size=32
# train_num_workers=1
# eval_batch_size=32
# eval_num_workers=1
# train_params = {'batch_size': train_batch_size, 'shuffle': True, 'num_workers': train_num_workers}
# train_loader = DataLoader(train_dataset, **train_params)
#
# eval_params = {'batch_size': eval_batch_size, 'shuffle': True, 'num_workers': eval_num_workers}
# eval_loader = DataLoader(eval_dataset, **eval_params)
#
#
# # define optimizer
# learning_rate=1e-3
# optimizer = AdamW(model.parameters(), lr=learning_rate)
#
# # define learning rate scheduler
# num_train_epochs = 3
# num_update_steps_per_epoch = len(train_loader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch
#
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )
#
# # define evaluation matrix
# rouge_score = evaluate.load("rouge")
#
# # training and evaluate
# progress_bar = tqdm(range(num_train_epochs))
# for epoch in range(num_train_epochs):
#     train_loss = 0
#     nb_train_steps = 0
#     nb_train_examples = 0
#     # training
#     model.train()
#     for batch in train_loader:
#         optimizer.zero_grad()
#
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#
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
#     print(f"Epoch {epoch}, Training Loss: {train_epoch_loss:.2f}")
#
#     eval_loss = 0
#     nb_eval_steps = 0
#     nb_eval_examples = 0
#
#     model.eval()
#     for batch in eval_loader:
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#         eval_loss += loss.item()
#
#         nb_eval_steps += 1
#         nb_eval_examples += len(batch)
#
#         ids = batch['input_ids']
#         mask = batch['attention_mask']
#
#         #max_length
#         generate_token = model.generate(ids, attention_mask=mask, max_new_tokens=512)
#         decoded_generate = tokenizer.batch_decode(generate_token, skip_special_tokens=True)
#
#         labels = batch["labels"]
#         labels = labels.numpy()
#
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#         decoded_generate, decoded_labels = postprocess_text(
#             decoded_generate, decoded_labels
#         )
#
#         rouge_score.add_batch(predictions=decoded_generate, references=decoded_labels)
#
#     eval_epoch_loss = eval_loss / nb_eval_steps
#     print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss:.2f}")
#
#     result = rouge_score.compute()
#     # Extract the median ROUGE scores
#     result = {key: value * 100 for key, value in result.items()}
#     result = {k: round(v, 4) for k, v in result.items()}
#     print(f"Epoch {epoch} rouge score: {result}")
#
#     progress_bar.update(1)
#
#     # check early stopping
#     check_early_stop = earlystop(train_loss, eval_loss)
#     if check_early_stop.early_stop:
#         model.save_pretrained(save_check_point_model + "early_stop")
#         break
#
#     # save check point
#     if epoch != 0 and epoch % 5 == 0:
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'learning_rate_schedular': lr_scheduler.state_dict()
#         }, save_check_point_model + "event_domain_adaption" + "epoch_" + str(epoch))
#
#         model.save_pretrained(save_check_point_model + "event_domain_adaption")

def domain_adaption_train_eval(json_filename,
                               model_checkpoint,
                               token_max_len,
                               domain_adaption,
                               wwm_prob,
                               train_eval_split,
                               train_batch_size,
                               eval_batch_size,
                               train_num_workers,
                               eval_num_workers,
                               num_train_epochs,
                               learning_rate,
                               save_check_point_model):

    with open(json_filename) as data_file:
        test_data = json.loads(data_file.read())

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, model_max_length=token_max_len)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    training_eval_dataset = summarization_data(test_data, tokenizer, domain_adaption=domain_adaption, wwm_prob=wwm_prob)
    print(f"number of entries in the dataset {len(training_eval_dataset)}")

    train_size = int(train_eval_split[0] * len(training_eval_dataset))
    test_size = len(training_eval_dataset) - train_size

    train_dataset, eval_dataset = random_split(training_eval_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    train_params = {'batch_size': train_batch_size, 'shuffle': True, 'num_workers': train_num_workers}
    train_loader = DataLoader(train_dataset, collate_fn=data_collator, **train_params)

    eval_params = {'batch_size': eval_batch_size, 'shuffle': True, 'num_workers': eval_num_workers}
    eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, **eval_params)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_update_steps_per_epoch = len(train_loader)
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

        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0

        model.eval()
        for batch in eval_loader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += len(batch)

            ids = batch['input_ids']
            mask = batch['attention_mask']

            # max_length
            generate_token = model.generate(ids, attention_mask=mask, max_new_tokens=512)
            decoded_generate = tokenizer.batch_decode(generate_token, skip_special_tokens=True)

            labels = batch["labels"]
            labels = labels.numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_generate, decoded_labels = postprocess_text(
                decoded_generate, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_generate, references=decoded_labels)

        eval_epoch_loss = eval_loss / nb_eval_steps
        print(f"Epoch {epoch}, Evaluation Loss: {eval_epoch_loss:.2f}")

        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch} rouge score: {result}")

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
            }, save_check_point_model + "event_domain_adaption" + "epoch_" + str(epoch))

            model.save_pretrained(save_check_point_model + "event_domain_adaption")

if __name__=="__main__":
    # load data
    json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'

    # tokenizer
    model_checkpoint = "t5-small"
    token_max_len = 512
    domain_adaption = True
    wwm_prob = 0.1

    #data loader
    train_eval_split=[0.7,0.3]
    train_batch_size = 32
    train_num_workers = 1
    eval_batch_size = 32
    eval_num_workers = 1


    #train
    learning_rate = 1e-3
    num_train_epochs = 3


    #save check point
    save_check_point_model = '../experiments/model_bucket/domain_adaption/t5/'

    domain_adaption_train_eval(json_filename,
                               model_checkpoint,
                               token_max_len,
                               domain_adaption,
                               wwm_prob,
                               train_eval_split,
                               train_batch_size,
                               eval_batch_size,
                               train_num_workers,
                               eval_num_workers,
                               num_train_epochs,
                               learning_rate,
                               save_check_point_model)