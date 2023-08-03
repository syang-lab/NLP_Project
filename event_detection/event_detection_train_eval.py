def event_detection_train_eval(args):
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
    train_eval_dataset = event_detection_data(train_eval_data, tokenizer, max_len=512, domain_adaption=False, wwm_prob=0.1)

    
    logger.debug(f"total number of entries in the dataset {len(train_eval_dataset)}")
    
    train_eval_split = [0.7, 0.3]
    train_size = int(train_eval_split[0] * len(train_eval_dataset))
    test_size = len(train_eval_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_eval_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    
    # pass data to dataloader
    train_params = {'batch_size': args.train_batch_size, 'shuffle': True, 'num_workers': args.train_num_workers}
    train_loader = DataLoader(train_dataset, **train_params)
    eval_params = {'batch_size': args.eval_batch_size, 'shuffle': True, 'num_workers': args.eval_num_workers}
    eval_loader = DataLoader(eval_dataset, **eval_params)

    
    # initialize model
    if not os.path.isdir(args.checkpoint_path + args.checkpoint_model):   
        logger.info(f" load hugging face model {args.checkpoint_model}")
        model = DistillBERTClass(args.checkpoint_model)
    else:
        logger.info(f" load pretrained model in pretrained model path: {args.checkpoint_path + args.checkpoint_model}")
        model = DistillBERTClass(args.checkpoint_path + args.checkpoint_model)
        
    # send to GPU
    model.to(device)
    
    # define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # define schedular
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs * len(train_loader),
    )
    
    
    # define load check point model
    if not os.path.isfile(args.checkpoint_path + "/event_domain_checkpoint.pt"):
        epoch_number = 0
    else:
        logger.info(f"load checkpoint model")
        checkpoint = torch.load(args.checkpoint_path + "/event_domain_checkpoint.pt")
        epoch_number = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['learning_rate_schedular'])  

     
    # train model
    logger.info(f"start training")
    progress_bar = tqdm(range(args.num_train_epochs))
    for epoch in range(epoch_number,args.num_train_epochs):
        train_loss = 0
        nb_train_correct = 0
        nb_train_steps = 0
        nb_train_examples = 0

        model.train()
        for _, data in enumerate(train_loader):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

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
        
        print(f"epoch {epoch}")
        print(f"training_loss={train_epoch_loss:.2f}")
        print(f"training_accuracy={train_epoch_accu:.2f}")

        # model evaluation
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_correct = 0
        nb_eval_examples = 0

        model.eval()
        for _, data in enumerate(eval_loader):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

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
        
        
        print(f"evaluation_loss={eval_epoch_loss:.2f}")
        print(f"evaluation_accuracy={eval_epoch_accu:.2f}")
        
        progress_bar.update(1)

        
        # check early stopping
        check_early_stop = earlystop(train_loss, eval_loss)
        if check_early_stop.early_stop:
            logger.info("save early_stop model")
            torch.save({'model_state_dict': model.state_dict()}, args.save_model + f"event_domain_early_stop_{epoch}.pt")
            break

            
        # save check point
        if epoch % 3 == 0:
            logger.info("save interval check point")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate_schedular': lr_scheduler.state_dict()
            }, args.checkpoint_path + "/event_domain_checkpoint.pt")

    
    logger.info("save final model")
    torch.save({'model_state_dict': model.state_dict()}, args.save_model + "/event_domain_final.pt")
    return



if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath('./code_event/'))
    
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    
    import json
    import torch
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from transformers import get_scheduler
    from event_detection_dataset import *
    from event_detection_model import *
    from early_stop import *
    
    
    import argparse
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument("--checkpoint_model", type=str, default='distilbert-base-cased')

    parser.add_argument("--train_batch_size", type=int, default=8)

    parser.add_argument("--eval_batch_size", type=int, default=8)

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
    
    
    event_detection_train_eval(parser.parse_args())