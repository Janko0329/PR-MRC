import time
common = {
    "data_home": "../../read_data/data/Data_set",
    "run_type": "train",  # train,test
    # "run_type": "test",  # train,test
}

train_config = {
    "train_data": "train.tsv",
    "valid_data": "dev.tsv",
    "test_data": "test.tsv",
    "path_to_save_model": "../../save_best_models",
    "hyper_parameters": {
        "lr": 1e-5,
        "batch_size": 22,
        "epochs": 100,
        "seed": 42,
        "max_seq_len": 512,
        "scheduler": "bioner_dataset"
    }
}

eval_config = {
    "dropout_prob":0.1,
    "hyper_parameters": {
        "batch_size": 32,
        "max_seq_len": 512,
    }
}

cawr_scheduler = {
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------------------------
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common}
eval_config = {**eval_config, **common}


import argparse
bert_model_path="../../pretrained_models/biobert_base"

from Code.run_ner.data_set_config import data_set_,ent2id
def bert_globalpointer_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)
    parser.add_argument("--method", default="bert_globalpointer", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    args = parser.parse_args()
    return args

def bert_prefix_globalpointer_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)
    parser.add_argument("--method", default="bert_prefix_globalpointer", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--prefix_projection", default=False)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--dropout_prob", default=0.1)
    parser.add_argument("--pre_seq_len", default=11)
    args = parser.parse_args()
    return args

def bert_GlobalPointer_regularization_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)
    parser.add_argument("--method", default="bert_globalpointer_regularization", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)

    parser.add_argument("--dropout_prob", default=0.1, type=int)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--fit_type", default="reg", type=str)
    parser.add_argument("--dis_type", default="KL", type=str)
    args = parser.parse_args()
    return args

def bert_prefix_GlobalPointer_regularization_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)
    parser.add_argument("--method", default="bert_prefix_globalpointer_regularization", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--pre_seq_len", default=11, type=int)
    parser.add_argument("--prefix_projection", default=False, type=str)

    parser.add_argument("--dropout_prob", default=0.1, type=int)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--fit_type", default="reg", type=str)
    parser.add_argument("--dis_type", default="KL", type=str)
    args = parser.parse_args()
    return args

def bert_efficient_globalpointer_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)
    parser.add_argument("--method", default="bert_efficient_globalpointer", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    args = parser.parse_args()
    return args

def bert_prefix_efficient_globalpointer_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)
    parser.add_argument("--method", default="bert_prefix_efficient_globalpointer", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--prefix_projection", default=False)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--dropout_prob", default=0.1)
    parser.add_argument("--pre_seq_len", default=11)
    args = parser.parse_args()
    return args

def bert_efficient_GlobalPointer_regularization_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)
    parser.add_argument("--method", default="bert_efficient_globalpointer_regularization", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)

    parser.add_argument("--dropout_prob", default=0.1, type=int)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--fit_type", default="reg", type=str)
    parser.add_argument("--dis_type", default="KL", type=str)
    args = parser.parse_args()
    return args

def bert_prefix_efficient_GlobalPointer_regularization_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)
    parser.add_argument("--method", default="bert_prefix_efficient_globalpointer_regularization", type=str)
    parser.add_argument("--DATA_SET_NAME", default=data_set_, type=str)
    parser.add_argument("--pre_seq_len", default=11, type=int)
    parser.add_argument("--prefix_projection", default=False, type=str)

    parser.add_argument("--dropout_prob", default=0.1, type=int)
    parser.add_argument("--ent_type_size", default=len(ent2id), type=int)
    parser.add_argument("--fit_type", default="reg", type=str)
    parser.add_argument("--dis_type", default="KL", type=str)
    args = parser.parse_args()
    return args





