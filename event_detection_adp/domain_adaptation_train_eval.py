def domain_adaption_train_eval(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    logger.info("check whether check point exist")
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
    
    
    # load file
    train_eval_filename=os.path.join(args.train_eval_filename, "train.json")
  
    with open(train_eval_filename) as data_file:
        train_eval_data = json.loads(data_file.read())

        
    # load training, validation and test data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=True)
    training_eval_dataset = event_detection_data(train_eval_data, tokenizer, max_len=512,
                                                 domain_adaption=True, wwm_prob=0.1)
    
    
    logger.info(f"total number of entries in the dataset {len(training_eval_dataset)}")
    
    train_eval_split=[0.7,0.3]
    train_size = int(train_eval_split[0] * len(training_eval_dataset))
    test_size = len(training_eval_dataset) - train_size

    train_dataset, eval_dataset = random_split(training_eval_dataset, [train_size,test_size],
                                               generator=torch.Generator().manual_seed(42))

    # pass data to dataloader
    train_params = {'batch_size': args.train_batch_size, 'shuffle': True, 'num_workers': args.train_num_workers}
    train_loader = DataLoader(train_dataset, **train_params)

    eval_params = {'batch_size': args.eval_batch_size, 'shuffle': True, 'num_workers': args.eval_num_workers}
    eval_loader = DataLoader(eval_dataset, **eval_params)

    # 2. construct model
    model = AutoModelForMaskedLM.from_pretrained('distilbert-base-cased')
    # set to GPU
    model.to(device)


    # 3. optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4. learning rate schedular
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs * len(train_loader),
    )

    
    #define load check point model
    if not os.path.isfile(args.checkpoint_path + "/event_domain_adaption_checkpoint.pt"):
        epoch_number = 0
    else:
        logger.info(f"load checkpoint model")
        checkpoint = torch.load(args.checkpoint_path + "/event_domain_adaption_checkpoint.pt")
        epoch_number = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['learning_rate_schedular'])  
    
    
    # train model
    logger.info(f"start training")
    progress_bar = tqdm(range(0,args.num_train_epochs))
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

        
        # evaluation
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

        eval_epoch_loss = eval_loss / nb_eval_steps
        
        print(f"evaluation_loss={eval_epoch_loss:.2f}")

        try:
            eval_perplexity = math.exp(eval_epoch_loss)
        except OverflowError:
            eval_perplexity = float("inf")
        

        print(f"evaluation_perplexity={eval_perplexity:.2f}")

        progress_bar.update(1)

        
        # check early stopping
        check_early_stop = earlystop(train_loss, eval_loss)
        if check_early_stop.early_stop:
            logger.info("save early_stop model")
            model.save_pretrained(args.save_model + "/event_domain_adaption_early_stop"+str(epoch))
            break

            
        # save check point
        if epoch % 5 == 0:
            logger.info("save interval check point")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate_schedular': lr_scheduler.state_dict()
            }, args.checkpoint_path + "/event_domain_adaption_checkpoint.pt")

    
    logger.info("save final model")
    model.save_pretrained(args.checkpoint_path + "/event_domain_adaption_final")
    return


if __name__=="__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath('./code/'))
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    
    import json
    import math
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from transformers import get_scheduler
    from event_detection_dataset import *
    from event_detection_model import *
    from early_stop import *
    
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size", type=int, default=16)

    parser.add_argument("--eval_batch_size", type=int, default=16)

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
    
    
    
    