import os
import set_config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel
from common.utils import Preprocessor, multilabel_categorical_crossentropy
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator,EfficientGlobalPointer,Encoder_Bert_prefix,create_regularization_GlobalPointer,create_regularization_EfficientGlobalPointer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
from evaluate import evaluate
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#获取实体类型数
from Code.read_data.read_file import read_ner_data_span
# from Code_GlobalPointer.GlobalPointer_pytorch.datasets.read_file import read_ner_data_span
def data_generator(data_type="train"):
    # 读取数据，生成DataLoader。
    if data_type == "train":
        train_data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        train_data_=read_ner_data_span(train_data_path, tokenizer, max_seq_length=512)

        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data_=read_ner_data_span(valid_data_path, tokenizer, max_seq_length=512)

        test_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        test_data_ = read_ner_data_span(test_data_path, tokenizer, max_seq_length=512)

    elif data_type == "valid":
        test_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        test_data = read_ner_data_span(test_data_path, tokenizer, max_seq_length=512)

    all_data = train_data_+valid_data_+test_data_

    # TODO:句子截取
    max_tok_num =0
    max_tok_num = max([len(f["input_ids"]) for f in all_data])
    assert max_tok_num <= hyper_parameters["max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])  #

    data_maker = DataMaker(tokenizer)

    if data_type == "train":
        train_dataloader = DataLoader(
                                      MyDataset(train_data_),
                                      # train_data_,
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id),
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data_),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        test_dataloader = DataLoader(MyDataset(test_data_),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=False,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return train_dataloader, valid_dataloader,test_dataloader #返回 train和dev数据

    elif data_type == "valid":
        valid_dataloader = DataLoader(MyDataset(test_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return valid_dataloader #返回 dev数据


metrics = MetricsCalculator() #评价指标


def loss_fun(y_true, y_pred):  # 损失函数
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)  # 多标签分类损失函数
    return loss

def train_step(batch_train, model, optimizer, criterion): #criterion损失函数
    # batch_input_ids:(batch_size, seq_len)
    # batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token = batch_train
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device),
                                                                                 batch_label_token.to(device)
                                                                                 )
    from set_config import regularization_GlobalPointer_EfficientGlobalPointer_config
    args = regularization_GlobalPointer_EfficientGlobalPointer_config()
    if args.use_regularization: #使用 正则化
        loss,logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels, criterion,batch_label_token)  # [btz, head, seq, seq]
    else: #不用 正则化
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids) #[btz, head, seq, seq]
        loss = criterion(logits, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(model, dataloader, epoch, optimizer):
    model.train()

    #scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]  #hyper_parameters["T_mult"]=1
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"] #2
        #线性预热
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,len(train_dataloader) * rewarm_epoch_num,T_mult)

    elif hyper_parameters["scheduler"] == "Step": #step_scheduler模式的参数 eg "decay_rate": 0.999,"decay_steps": 100
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader)) #枚举dataloader

    total_loss = 0.
    for batch_ind, batch_data in pbar:
        loss= train_step(batch_data, model, optimizer, loss_fun) #调用train_step 得到当前训练步数的损失

        total_loss += loss #总损失

        avg_loss = total_loss / (batch_ind + 1)
        scheduler.step()

        #设置/修改进度条的描述
        pbar.set_description(f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        # 设置 / 修改(附加属性)
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

def valid_test_step(batch_valid, model):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels ,batch_label_token= (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device),
                                                                                 batch_label_token.to(device)
                                                                                 )
    with torch.no_grad():
        from set_config import regularization_GlobalPointer_EfficientGlobalPointer_config
        args = regularization_GlobalPointer_EfficientGlobalPointer_config()
        if args.use_regularization:  # 使用 正则化
            _,logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels, loss_fun, batch_label_token) #模型的输出
        else:
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids) #模型的输出

    sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels) #评价得分

    return sample_f1, sample_precision, sample_recall #返回结果

def valid(model, dataloader):
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader, desc="deving"): #枚举dev数据
        f1, precision, recall = valid_test_step(batch_data, model) #调用valid_step函数 得到当前批次数据的 评价得分

        total_f1 += f1
        total_precision += precision
        total_recall += recall

    #取均值
    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))

    print("******************************************")
    print('dev:  '+f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    return avg_f1

def test(model, dataloader):
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader, desc="testing"): #枚举dev数据
        f1, precision, recall = valid_test_step(batch_data, model) #调用valid_step函数 得到当前批次数据的 评价得分

        total_f1 += f1
        total_precision += precision
        total_recall += recall

    #取均值
    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))

    print('test:  '+f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")
    print()
    return avg_f1

if __name__ == '__main__':
    config = set_config.train_config
    hyper_parameters = config["hyper_parameters"]  # 超参
    config["num_workers"] = 0 if sys.platform.startswith("linux") else 0
    torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)

    from set_config import commen_config
    args = commen_config()
    args.n_gpu = torch.cuda.device_count()

    NCBI_BC5CDR_dise_SPAN_LABEL_TO_ID = {'O': 0, 'Disease': 1} #首先将标签为Disease的 转为id的形式
    ent2id={"0":0,"1":1}


    ################ 1 ############## regularization_GlobalPointer_config use_regularization设置为False

    # encoder = BertModel.from_pretrained(config["bert_path"])  # 用bert作为encoder
    # encoder = Encoder_Bert_prefix(config["bert_path"])  # 用bert+prefix作为encoder regularization_GlobalPointer_config use_prefix设置为True

    # model = GlobalPointer(encoder, args.ent_type_size, 64)  # GlobalPointer模型
    # model = EfficientGlobalPointer(encoder, args.ent_type_size, 64)  # EfficientGlobalPointer模型

    #################################


    ################# 2 ################ regularization_GlobalPointer_config use_regularization设置为True

    model = create_regularization_GlobalPointer(args)  # GlobalPointer模型
    # model = create_regularization_EfficientGlobalPointer(args)  # EfficientGlobalPointer模型

    #################################

    model = model.to(device)

    if config["run_type"] == "train": #训练模式 进行训练和评估
        train_dataloader, valid_dataloader,test_dataloader = data_generator()

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

        max_f1 = 0.
        for epoch in range(hyper_parameters["epochs"]):
            print("===================epoch====================",epoch+1)
            train(model, train_dataloader, epoch, optimizer) #训练
            valid_f1 = valid(model, valid_dataloader) #评估
            test_f1 = test(model, test_dataloader)
            if test_f1 > max_f1:
                max_f1 = test_f1
                if test_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    torch.save(model.state_dict(),os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(model_state_num)))
            print(f"****************** Best F1 ******************: {max_f1}")
            print()

    elif config["run_type"] == "eval": #评估模式 只进行评估
        evaluate()

if (args1.use_prefix):  # 定义encoder
    encoder = Encoder_Bert_prefix(args.model_name_or_path[i])
else:
    from transformers import BertModel
    encoder = BertModel.from_pretrained(args.model_name_or_path[i])
