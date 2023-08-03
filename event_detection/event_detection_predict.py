import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from FTE_NLP.model.event_detection_dataset import *
from FTE_NLP.model.event_detection_model import *


def prediction(dev_filename,token_pre_trained_model,token_max_len,dev_batch_size,dev_num_workers,checkpoint_model):
    # load file
    with open(dev_filename) as data_file:
        dev_data = json.loads(data_file.read())

    tokenizer = AutoTokenizer.from_pretrained(token_pre_trained_model, use_fast=True)
    dev_dataset = event_detection_data(dev_data, tokenizer, max_len=token_max_len)

    dev_params = {'batch_size': dev_batch_size, 'shuffle': True, 'num_workers': dev_num_workers}
    dev_loader = DataLoader(dev_dataset, **dev_params)

    model = DistillBERTClass(checkpoint_model)

    nb_dev_correct = 0
    nb_dev_examples = 0
    model.eval()
    for _, data in enumerate(dev_loader):
        ids = data['input_ids']
        mask = data['attention_mask']
        targets = data['targets']

        with torch.no_grad():
            outputs = model(ids, mask)

        max_val, max_idx = torch.max(outputs.data, dim=1)
        nb_dev_correct += (max_idx == targets).sum().item()

        nb_dev_examples += targets.size(0)

    eval_epoch_accu = (nb_dev_correct * 100) / nb_dev_examples
    print(f"Evaluation Accuracy: {eval_epoch_accu}")


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # input path
    dev_filename = '../data/raw_EDT/Event_detection/dev.json'

    # pretrained model
    token_pre_trained_model = 'distilbert-base-cased'

    # tokenizer
    token_max_len = 512
    domain_adaption = False
    mask_prob = 0

    # load model
    checkpoint_model='distilbert-base-cased'
    # checkpoint_model = '../experiments/model_bucket/domain_adaption/distilbert/event_domain_adaption'

    # prediction
    dev_batch_size = 32
    dev_num_workers = 1

    prediction(dev_filename, token_pre_trained_model, token_max_len, dev_batch_size, dev_num_workers, checkpoint_model)