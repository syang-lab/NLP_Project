def domain_adaption_train_eval(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    
    logger.info("check whether check point exist")
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
    
    
    # load data file
    train_eval_filename=os.path.join(args.train_eval_filename, "evaluate_news.json")
    with open(train_eval_filename) as data_file:
        data = json.loads(data_file.read())
    
    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # construct dataset
    training_eval_dataset = summarization_data(data[0:3000], tokenizer, domain_adaption=True, wwm_prob=0.1)
    logger.debug(f"total number of entries in the dataset {len(training_eval_dataset)}")
    
    
    train_eval_split=[0.7,0.3]
    train_size = int(train_eval_split[0] * len(training_eval_dataset))
    test_size = len(training_eval_dataset) - train_size
    train_dataset, eval_dataset = random_split(training_eval_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))
    
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    train_params = {'batch_size': args.train_batch_size, 'shuffle': True, 'num_workers': args.train_num_workers}
    train_loader = DataLoader(train_dataset, collate_fn=data_collator, **train_params)
    eval_params = {'batch_size': args.eval_batch_size, 'shuffle': True, 'num_workers': args.eval_num_workers}
    eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, **eval_params)

    
    # send load model
    model = model.to(device)
    
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    
    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    
    # define load check point model
    if not os.path.isfile(args.checkpoint_path + "/summarization_domain_adaption_checkpoint.pt"):
        epoch_number = 0
    else:
        logger.info(f"load checkpoint model")
        checkpoint = torch.load(args.checkpoint_path + "/summarization_domain_adaption_checkpoint.pt")
        epoch_number = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['learning_rate_schedular'])  
    
    
    # define evaluation matrix
    rouge_score = evaluate.load("rouge")

    
    # training and evaluate
    logger.info(f"start training")
    progress_bar = tqdm(range(args.num_train_epochs))
    for epoch in range(epoch_number, args.num_train_epochs):
        train_loss = 0
        nb_train_steps = 0
        nb_train_examples = 0
        # training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            # move item to GPU
            for key, value in batch.items():
                batch[key] = batch[key].to(device)
                
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            nb_train_steps += 1
            nb_train_examples += len(batch)

            lr_scheduler.step()

        train_epoch_loss = train_loss / nb_train_steps
        print(f"epoch {epoch}")
        print(f"training_loss={train_epoch_loss:.2f}")
        
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0

        model.eval()
        for batch in eval_loader:
            # move item to GPU
            for key, value in batch.items():
                batch[key] = batch[key].to(device)
            
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
            decoded_generate = tokenizer.batch_decode(generate_token.cpu(), skip_special_tokens=True)

            labels = batch["labels"].cpu()
            labels = labels.numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_generate, decoded_labels = postprocess_text(
                decoded_generate, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_generate, references=decoded_labels)

        eval_epoch_loss = eval_loss / nb_eval_steps
        print(f"evaluation_loss={eval_epoch_loss:.2f}")
        
        
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"rouge1={result['rouge1']:.2f}")
        print(f"rouge2={result['rouge2']:.2f}")
        print(f"rougeL={result['rougeL']:.2f}")
        print(f"rougeLsum={result['rougeLsum']:.2f}")
        
        
        progress_bar.update(1)

        # check early stopping
        check_early_stop = earlystop(train_loss, eval_loss)
        if check_early_stop.early_stop:
            logger.info("save early_stop model")
            model.save_pretrained(args.save_model + "/summarization_domain_adaption_early_stop"+str(epoch))
            break

        # save check point
        if epoch % 5 == 0:
            logger.info("save interval check point")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate_schedular': lr_scheduler.state_dict()
            }, args.checkpoint_path + "/summarization_domain_adaption_checkpoint.pt")

    
    logger.info("save final model")
    model.save_pretrained(args.checkpoint_path + "/summarization_domain_adaption_final")
    return

    
if __name__=="__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath('./code_summary_adp/'))
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    import nltk
    nltk.download('punkt')
    
    import numpy as np
    import evaluate
    import rouge_score
    from tqdm.auto import tqdm
    from transformers import T5Tokenizer
    from transformers import AutoModelForSeq2SeqLM
    from transformers import DataCollatorForSeq2Seq
    from torch.utils.data import DataLoader, random_split
    from summarization_dataset import *
    from torch.optim import AdamW
    from transformers import get_scheduler
    from post_process import *
    from early_stop import *
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    
    # Container environment
    parser.add_argument("--train_eval_filename", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    # save the model to current position in the docker
    parser.add_argument("--save_model", type=str, default=os.environ["SM_MODEL_DIR"])
                              
    # the path of model copied from S3 
    parser.add_argument("--checkpoint_path",type=str,default="/opt/ml/checkpoints")
    domain_adaption_train_eval(parser.parse_args())