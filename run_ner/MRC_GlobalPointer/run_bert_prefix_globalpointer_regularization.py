import os
import sys
import torch
from transformers import BertTokenizerFast, BertModel
from Code.read_data.utils import Preprocessor
from Code.models.global_method import DataMaker, MyDataset, GlobalPointer, MetricsCalculator,EfficientGlobalPointer,Encoder_Bert_prefix,create_bert_prefix_GlobalPointer_regularization
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from Code.read_data.read_file import read_ner_data_span
def data_generator(data_type="train"):
    # 读取数据，生成DataLoader。
    if data_type == "train":
        train_data_path = os.path.join(config["data_home"], args.DATA_SET_NAME, config["train_data"])
        train_data_=read_ner_data_span(train_data_path, tokenizer, max_seq_length=512)

        valid_data_path = os.path.join(config["data_home"], args.DATA_SET_NAME, config["valid_data"])
        valid_data_=read_ner_data_span(valid_data_path, tokenizer, max_seq_length=512)

        test_data_path = os.path.join(config["data_home"], args.DATA_SET_NAME, config["test_data"])
        test_data_ = read_ner_data_span(test_data_path, tokenizer, max_seq_length=512)

    elif data_type == "valid":
        test_data_path = os.path.join(config["data_home"], args.DATA_SET_NAME, config["test_data"])
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
from Code.models.losses.global_pointer_loss import loss_fun

def train_step(batch_train, model, optimizer, criterion): #criterion损失函数
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token = batch_train
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device),
                                                                                 batch_label_token.to(device)
                                                                                 )
    loss, logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels, criterion,
                         batch_label_token)  # [btz, head, seq, seq]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(model, dataloader, epoch, optimizer):
    # if os.path.exists(model_state_dir):
    #     print("=================loading_model_for_training===================")
    #     # checkpoint = torch.load(model_state_dir, map_location='cpu')  # 设置断点 断点续训
    #     # model.load_state_dict(checkpoint['model_state_dict'])  # 加载断点的模型
    #     model.load_state_dict(torch.load(model_state_dir))
    # else:
    #     print("\n==================start_training===================")
    model.train()

    #scheduler
    if hyper_parameters["scheduler"] == "bioner_dataset":
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
        pbar.set_description(f'Project:{args.DATA_SET_NAME}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
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
        _, logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels, loss_fun,
                          batch_label_token)  # 模型的输出

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
    info_dev = 'dev:  '+f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}'
    print(info_dev)
    return avg_f1,avg_precision,avg_recall,info_dev

def test(model, dataloader):
    # if os.path.exists(model_state_dir):
    #     print("=================loading_model_for_testing===================")
    #     model.load_state_dict(torch.load(model_state_dir))
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

    info_test = 'test:  '+f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}'
    print(info_test)
    print("******************************************")
    print()
    return avg_f1,avg_precision,avg_recall,info_test

if __name__ == '__main__':
    from Code.run_ner import MRC_EG_G_Set_Config
    config = MRC_EG_G_Set_Config.train_config
    hyper_parameters = config["hyper_parameters"]  # 超参
    config["num_workers"] = 0 if sys.platform.startswith("linux") else 0
    torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed

    torch.backends.cudnn.deterministic = True

    from Code.run_ner.MRC_EG_G_Set_Config import bert_prefix_GlobalPointer_regularization_config
    args = bert_prefix_GlobalPointer_regularization_config()
    args.n_gpu = torch.cuda.device_count()
    from Code.run_ner.MRC_EG_G_Set_Config import ent2id


    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path[0], add_special_tokens=True, do_lower_case=False)
    model = create_bert_prefix_GlobalPointer_regularization(args)
    model = model.to(device)

    train_dataloader, valid_dataloader,test_dataloader = data_generator()

    init_learning_rate = float(hyper_parameters["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    max_f1 = 0.
    from Code.read_data.save import save,mkdir
    temp_path1 = config["path_to_save_model"]+"/"+args.DATA_SET_NAME+"/"+args.method+"/"
    temp_path2 = "lr_"+str(hyper_parameters["lr"])+"_"+"seed_"+str(hyper_parameters["seed"])
    temp_path3 = os.path.join(temp_path1,temp_path2)
    mkdir(temp_path3)

    model_state_dir = os.path.join(temp_path3, "pytorch_model.bin")
    write_training_info = os.path.join(temp_path3, "training_info.txt")
    write_best_score_path = os.path.join(temp_path3, "best_score.txt")

    if config["run_type"] == "train": #train模式 训练 评估 测试
        for epoch in range(hyper_parameters["epochs"]):
            print("===================epoch====================",epoch+1)
            train(model, train_dataloader, epoch, optimizer) #训练
            valid_f1,valid_p,valid_r,valid_info = valid(model, valid_dataloader) #评估
            test_f1,test_p,test_r,test_info = test(model, test_dataloader)

            with open(write_training_info, 'a+') as file1: #保存训练信息
                epoch_string = str(epoch+1)+"\n"
                file1.write(epoch_string)
                file1.write(valid_info)
                file1.write("\n")
                file1.write(test_info)
                file1.write("\n")

            if test_f1 > max_f1:
                max_f1 = test_f1
                save(model, optimizer, model_state_dir)  # 保存模型
                print(f"****************** Best F1 ******************: {max_f1}")
                print()

                with open(write_best_score_path, 'w+') as file2: #保存最好情况下的得分
                    string1 = "dev" + "\n" + "F1:" + str(valid_f1) + "\n" + "best_presion:" + str(valid_p) + "\n" + "recall:" + str(valid_r)
                    file2.write(string1)
                    file2.write("\n\n")
                    string2 = "test" + "\n" + "F1:" + str(test_f1) + "\n" + "best_presion:" + str(test_p) + "\n" + "recall:" + str(test_r)
                    file2.write(string2)

    if config["run_type"] == "test": #只进行测试
            test(model, test_dataloader)

