import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import os
import argparse
from FTE_NLP.train_eval_pred.domain_adaptation_train_eval import *
from FTE_NLP.utils.domain_adaption_train_eval_default_config import *
from FTE_NLP.utils.save_model import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_eval_config_dir_filename',
        type=str,
        required=True,
        help='train and eval config filename')

    parser.add_argument(
        '--train_eval_data_directory',
        type=str,
        help='directory for train and evaluation data'
    )

    parser.add_argument(
         "--cloud_directory",
         type=str,
         required=True,
         default='project-nlp-375001-aiplatform',
         help="directory where model checkpoints, logs and other artifacts are saved."
    )

    return parser.parse_args()


def populate_data_dir(dataloader_cfg, data_dir):
    dataloader_cfg.defrost()
    dataloader_cfg.Train_Eval_Filename = data_dir
    dataloader_cfg.freeze()
    return dataloader_cfg


def main():
    args = parse_args()
    cfg = load_config(args.train_eval_config_dir_filename)

    dataloader_config = cfg.DATALOADER
    dataloader_config = populate_data_dir(dataloader_config, args.train_eval_data_directory)

    train_eval_config = cfg.TRAIN_EVAL
    domain_adaption_train_eval(dataloader_config.Train_Eval_Filename,
                               dataloader_config.Token_Pre_Trained_Model,
                               dataloader_config.Train_Eval_Split,
                               dataloader_config.Token_Max_Len,
                               dataloader_config.Domain_Adaption,
                               dataloader_config.Mask_Prob,
                               dataloader_config.Train_Batch_Size,
                               dataloader_config.Eval_Batch_Size,
                               dataloader_config.Train_Num_Workers,
                               dataloader_config.Eval_Num_Workers,
                               train_eval_config.Learning_Rate,
                               train_eval_config.Num_Train_Epochs,
                               train_eval_config.Save_Check_Point_Model)

    save_model(train_eval_config.Save_Check_Point_Model, args.cloud_directory)


if __name__ == "__main__":
# example code: python FTE_NLP/train_eval_pred/domain_adaption_main.py --train_eval_config_directory FTE_NLP/configs/
# --train_eval_config_filename domain_adaption_train_eval_config.yaml
# --train_eval_data_directory FTE_NLP/data/raw_EDT/Event_detection/dev_test.json

    main()
