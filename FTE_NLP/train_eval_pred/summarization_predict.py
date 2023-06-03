from tqdm.auto import tqdm
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, random_split
from FTE_NLP.model.summarization_dataset import *
from torch.optim import AdamW
from transformers import get_scheduler
from FTE_NLP.utils.post_process import *

import numpy as np
import evaluate


def prediction(json_filename,model_checkpoint,token_max_len,dev_batch_size,dev_num_workers):
    json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'
    with open(json_filename) as data_file:
        dev_data = json.loads(data_file.read())

    # model_checkpoint = "t5-small"
    # token_max_len=512
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, model_max_length=token_max_len)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    dev_dataset = summarization_data(dev_data, tokenizer, domain_adaption=False, wwm_prob=0.1)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    # dev_batch_size=32
    # dev_num_workers=1
    dev_params = {'batch_size': dev_batch_size, 'shuffle': True, 'num_workers': dev_num_workers}
    dev_loader = DataLoader(dev_dataset, collate_fn=data_collator, **dev_params)


    # define evaluation matrix
    rouge_score = evaluate.load("rouge")

    model.eval()
    for batch in dev_loader:
        with torch.no_grad():
            outputs = model(**batch)

        ids = batch['input_ids']
        mask = batch['attention_mask']

        #max_length
        generate_token = model.generate(ids, attention_mask=mask, max_new_tokens=50)
        decoded_generate = tokenizer.batch_decode(generate_token, skip_special_tokens=True)

        labels = batch["labels"]
        labels = labels.numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_generate, decoded_labels = postprocess_text(
            decoded_generate, decoded_labels
        )

        rouge_score.add_batch(predictions=decoded_generate, references=decoded_labels)

        # print context, prediction and label
        input_context = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        print(f" input context: {input_context}")
        print(f" predict summarization: {decoded_generate}")

    result = rouge_score.compute()
    # Extract the median ROUGE scores
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"rouge score: {result}")
    return

if __name__=="__main__":
    # load data
    json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news_test.json'

    # tokenizer
    model_checkpoint = "t5-small"
    token_max_len = 512

    #data loader
    dev_batch_size = 32
    dev_num_workers = 1

    prediction(json_filename, model_checkpoint, token_max_len, dev_batch_size, dev_num_workers)


