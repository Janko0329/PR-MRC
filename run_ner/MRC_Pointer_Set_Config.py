import argparse
from Code.run_ner.data_set_config import data_set_,crf_LABEL_TO_ID,SPAN_LABEL_TO_ID,LABEL_TO_ID

bert_model_path="../../pretrained_models/biobert_base"
def set_bert_crf_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--loss_type", type=str, default="CE") #CE FL LS DL

    parser.add_argument("--use_manual_template", default=False),  # True False
    parser.add_argument("--soft_prompt", default=False),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_crf/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(crf_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_lstm_crf_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--loss_type", type=str, default="CE") #CE FL LS DL

    parser.add_argument("--use_manual_template", default=False),  # True False
    parser.add_argument("--soft_prompt", default=False),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--batch_size", default=52, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_lstm_crf/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(crf_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_softmax_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--loss_type", type=str, default="FL") #CE FL LS DL

    parser.add_argument("--use_manual_template", default=False),  # True False
    parser.add_argument("--soft_prompt", default=False),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_softmax/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_span_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)
    # parser.add_argument("--model_name_or_path", default=["dmis-lab/biobert-large-cased-v1.1"], type=str)
    # parser.add_argument("--model_name_or_path", default=["dmis-lab/biobert-large-cased-v1.1-squad"], type=str)

    parser.add_argument("--span_ner",default=True)
    parser.add_argument("--loss_type", type=str, default="CE") #CE FL LS DL

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--use_manual_template", default=False),  # True False
    parser.add_argument("--soft_prompt", default=False),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models_test/"+data_set_+"/bert_span/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(SPAN_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_span_regularization():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_ter_name_or_path", default=["dmis-lab/biobert-v1.1"], type=str)
    # parser.add_argument("--model_stu_name_or_path", default=["albert-base-v2"], type=str)
    # parser.add_argument("--model_stu_name_or_path", default=["prajjwal1/bert-tiny"], type=str)
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)

    parser.add_argument("--loss_type", type=str, default="CE") #CE FL LS DL
    parser.add_argument('--dis_type',default="KL",type=str)
    parser.add_argument('--fit_type',default="reg",type=str) #rg  mean

    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--span_ner",default=True) #bert span read file

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_span_regularization/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(SPAN_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_span_prefix_tuning_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--prefix_prompt", default=True),  #True False  prefix tuning
    parser.add_argument("--do_adv",default=False)
    parser.add_argument("--prefix_projection", default=False),  #True False
    parser.add_argument("--prefix_seq_len",type=int,default=11)

    parser.add_argument("--loss_type", type=str, default="CE")  # CE FL LS DL
    parser.add_argument("--span_ner",default=True)  #bert span read file

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_span_prefix_tuning/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(SPAN_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_span_prefix_regularization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)

    parser.add_argument("--have_ans_predict",default=False) #span+prefix 是否进行sentence_classification预测
    parser.add_argument('--do_adv',default=False)
    parser.add_argument("--prefix_prompt", default=True),  #True False
    parser.add_argument("--prefix_projection", default=False),  #True False
    parser.add_argument("--prefix_seq_len",type=int,default=11)

    parser.add_argument("--span_ner_have_ans",default=True) #读数据span+prefix+sentence_classification
    parser.add_argument("--loss_type", type=str, default="CE")  # CE FL LS DL
    parser.add_argument('--dis_type',default="KL",type=str)
    parser.add_argument('--fit_type',default="reg",type=str) #rg  mean

    parser.add_argument("--batch_size", default=6, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/bert_span_prefix_regularization_test/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(SPAN_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_span_prefix_tuning_regularization_have_ans():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=[bert_model_path,bert_model_path], type=str)

    parser.add_argument("--self_attention",default=False)
    parser.add_argument("--have_ans_predict",default=True) #span+prefix 是否进行sentence_classification预测
    parser.add_argument('--do_adv',default=False)
    parser.add_argument("--prefix_prompt", default=True),  #True False
    parser.add_argument("--prefix_projection", default=False),  #True False
    parser.add_argument("--prefix_seq_len",type=int,default=11)

    parser.add_argument("--span_ner_have_ans_att",default=True) #读数据span+prefix+sentence_classification
    parser.add_argument("--loss_type", type=str, default="CE")  # CE FL LS DL
    parser.add_argument('--dis_type',default="KL",type=str)
    parser.add_argument('--fit_type',default="reg",type=str) #rg  mean

    parser.add_argument("--batch_size", default=22, type=int)
    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/prefix_regularization_have_ans_att/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_NCBI')
    parser.add_argument("--num_class", type=int, default=len(SPAN_LABEL_TO_ID))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()
    return args

def set_bert_sapn_soft_prompt_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", default=["dmis-lab/biobert-v1.1","dmis-lab/biobert-v1.1"], type=str)
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--prefix_prompt", default=False),  #True False
    parser.add_argument("--prefix_projection", default=False),  #True False
    parser.add_argument("--prefix_seq_len",type=int,default=15)

    parser.add_argument("--use_manual_template", default=False),  # True False

    parser.add_argument("--soft_prompt", default=True),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--loss_type", type=str, default="FL") #CE FL LS DL
    parser.add_argument("--q_type", type=str, default="reg") # reg weight mean
    parser.add_argument("--dis_type", type=str, default='KL')  # KL BD JS ED CE WD HD

    parser.add_argument("--batch_size", default=52, type=int)

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/", type=str)
    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_dmis_crf_test')
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=len(LABEL_TO_ID))

    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)

    args = parser.parse_args()
    return args

def set_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", default=["dmis-lab/biobert-v1.1","dmis-lab/biobert-v1.1"], type=str)
    parser.add_argument("--model_name_or_path", default=[bert_model_path], type=str)

    parser.add_argument("--prefix_prompt", default=False),  #True False
    parser.add_argument("--prefix_projection", default=False),  #True False
    parser.add_argument("--prefix_seq_len",type=int,default=15)

    parser.add_argument("--use_manual_template", default=False),  # True False

    parser.add_argument("--soft_prompt", default=True),  #True False
    parser.add_argument("--soft_prompt_token", default=20, type=int)

    parser.add_argument("--loss_type", type=str, default="FL") #CE FL LS DL
    parser.add_argument("--q_type", type=str, default="reg") # reg weight mean
    parser.add_argument("--dis_type", type=str, default='KL')  # KL BD JS ED CE WD HD

    parser.add_argument("--batch_size", default=52, type=int)

    parser.add_argument("--model_dataset", type=str, default='dmis-lab-biobert-v1.1_dmis_crf_test')

    parser.add_argument("--train_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/train.tsv')
    parser.add_argument("--dev_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/dev.tsv')
    parser.add_argument("--test_file", type=str, default='../../read_data/data/Data_set/'+data_set_+'/test.tsv')

    parser.add_argument("--output_model_root_path", default="../../save_best_models/"+data_set_+"/", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=len(LABEL_TO_ID))

    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--alpha", type=float, default=50.0)

    args = parser.parse_args()
    return args