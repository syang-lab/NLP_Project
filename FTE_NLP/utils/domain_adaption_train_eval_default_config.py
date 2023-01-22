from yacs.config import CfgNode as CN

_C = CN()
_C.DATALOADER = CN()
_C.DATALOADER.Train_Eval_Filename = 'FTE_NLP/data/raw_EDT/Event_detection/dev_test.json'
_C.DATALOADER.Token_Pre_Trained_Model = 'distilbert-base-cased'
_C.DATALOADER.Domain_Adaption = 'True'
_C.DATALOADER.Train_Eval_Split = [0.7, 0.3]
_C.DATALOADER.Token_Max_Len = 512
_C.DATALOADER.Mask_Prob = 0.1
_C.DATALOADER.Train_Batch_Size = 64
_C.DATALOADER.Eval_Batch_Size = 64
_C.DATALOADER.Train_Num_Workers = 0
_C.DATALOADER.Eval_Num_Workers = 0

_C.TRAIN_EVAL = CN()
_C.TRAIN_EVAL.Learning_Rate = 1e-05
_C.TRAIN_EVAL.Num_Train_Epochs = 3
_C.TRAIN_EVAL.Save_Check_Point_Model = 'FTE_NLP/experiments/model_bucket/domain_adaption/distilbert/'
_C.TRAIN_EVAL.Cloud_Storage = ''

def get_cfg_defaults():
    return _C.clone()


def load_config(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
