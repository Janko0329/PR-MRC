import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from Code.models.learning_rate_scheduler import get_linear_schedule_with_warmup
from Code.read_data.model_input_feture import set_seed, collate_fn
from Code.read_data.read_file import read_ner_data
from Code.read_data.manual_template_making import read_manual_emplate_data,read_test_data
from Code.read_data.save import save,mkdir
from torch.cuda.amp import autocast, GradScaler
import seqeval.metrics
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
from Code.models.losses.diceLoss import DiceLoss
from Code.models.losses.focal_loss import FocalLoss
from Code.models.losses.label_smoothing import LabelSmoothingCrossEntropy

from Code.run_ner.MRC_Pointer_Set_Config import LABEL_TO_ID,crf_LABEL_TO_ID,SPAN_LABEL_TO_ID
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

class BertSoftmax(nn.Module):
    def __init__(self, args, model_name):
        super().__init__()
        self.args = args
        self.model_name_ = model_name
        self.num_labels = self.args.num_class

        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        self.model = AutoModel.from_pretrained(self.model_name_, )
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier2 = nn.Linear(config.hidden_size, args.num_class)
        self.loss_type = args.loss_type

    def forward(self,args, input_ids, attention_mask, labels=None):
        args=args
        outputs = self.model(input_ids, attention_mask, return_dict=False)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        log = self.classifier1(sequence_output)
        log = self.dropout(log)
        logits = self.classifier2(log) #[b s num_class]
        hiden = outputs[2:]  #[b s 768]
        outputs = (logits,) + hiden

        if labels is not None:
            if args.loss_type == "CE":
                self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)
            elif args.loss_type == "FL":
                self.loss_fnt = FocalLoss(ignore_index=-1)
                # self.loss_fnt = FocalLoss(gamma = 2, alpha = [Data_set.0] * 7)
            elif args.loss_type == "LS":
                self.loss_fnt = LabelSmoothingCrossEntropy(ignore_index=-1)
            elif args.loss_type == "DL":
                self.loss_fnt = DiceLoss(ignore_index=-1)

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fnt(active_logits, active_labels)
            else:
                loss = self.loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    if os.path.exists(output_model):
        print("=================loading_model_for_training===================")
        # checkpoint = torch.load(output_model, map_location='cpu')  # 设置断点 断点续训
        # model.load_state_dict(checkpoint['model_state_dict'])  # 加载断点的模型
        model.load_state_dict(torch.load(output_model))
    else:
        print("\n==================start_training===================")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    num_steps = 0
    best_f1=0
    for epoch in range(int(args.num_train_epochs)):
        current_epoch = epoch+1
        print("\n===================epoch====================   ", current_epoch)

        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:  #还在热身warmup阶段
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha  #下一阶段 引入一致性损失

            batch = {key: value.to(args.device) for key, value in batch.items()}  #将batch数据放到设备上
            with autocast():
                outputs = model(args,**batch) #将一批次数据输入模型中，得到输出

            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
            if step == len(train_dataloader) - 1:  #训练已达最后一步
                for tag, features in benchmarks:
                    best_f1 = evaluate(args, model, optimizer ,features, current_epoch ,tag=tag)
    return best_f1

def evaluate(args, model, optimizer ,features, current_epoch,tag="dev_or_test"):
    #保存训练信息
    with open(write_training_info, 'a+') as file1:
        str0 = "epoch:"+str(current_epoch)
        file1.write("\n")
        file1.write(str0)
        file1.write("\n")

    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []
    for batch in dataloader:
        model.eval() #将模型设置为评估模式

        # batch {[input_id] [attention_mask] labels}  {[b*s] [b*s] }
        batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()  #从batch数据中取出lable 转为列表格式 再转存到keys中

        batch['labels'] = None
        with torch.no_grad():
            logits = model(args,**batch)[0]
            # preds += np.argmax(logits.cpu().numpy(), axis=-Data_set).flatten().tolist()  # 将模型的预测结果 转存到preds上
            preds+= np.argmax(logits.cpu().numpy(), axis=2).flatten().tolist()

    preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))

    preds = [ID_TO_LABEL[pred] for pred in preds]
    keys = [ID_TO_LABEL[key] for key in keys] #将index转为key形式
    model.zero_grad()

    # metrics
    f1_ = seqeval.metrics.f1_score([keys], [preds])
    F1_Macro = seqeval.metrics.f1_score([keys], [preds],average='macro')
    F1_Micro = seqeval.metrics.f1_score([keys], [preds],average='micro')
    acc=seqeval.metrics.accuracy_score([keys], [preds])
    precision=seqeval.metrics.precision_score([keys], [preds])
    recall=seqeval.metrics.recall_score([keys], [preds])

    if tag=='dev':
        global temp_dev_f1_macro, temp_dev_f1_micro,temp_dev_f1,temp_dev_acc,temp_dev_pre,temp_dev_recall
        temp_dev_f1_macro = F1_Macro
        temp_dev_f1_micro = F1_Micro
        temp_dev_f1 = f1_
        temp_dev_acc = acc
        temp_dev_pre = precision
        temp_dev_recall = recall
        print('\n')

    print(tag)
    output = {"F1": f1_,"F1_Macro": F1_Macro,"F1_Micro": F1_Micro,"accuracy": acc, "precision": precision, "recall": recall}
    print(output)

    #将训练过程中 训练信息保存下来
    with open(write_training_info, 'a+') as file2:
        str1 = tag +"\t" + "F1:" + str(f1_) + "\t" + "F1_Macro:" + str(F1_Macro) + "\t" + "F1_Micro:" + str(F1_Micro) + "\t" + "accuracy:" + str(
            acc) + "\t" + "precision:" + str(precision) + "\t" + "recall:" + str(recall)
        file2.write(str1)
        file2.write("\n")

    if tag=='test':
        global best_score_F1_Macro,best_score_F1_Micro,best_f1,best_acc,best_pre,best_recall
        if(best_f1 < f1_):
            best_f1 = f1_
            best_score_F1_Macro = F1_Macro
            best_score_F1_Micro = F1_Micro
            best_acc = acc
            best_pre = precision
            best_recall = recall
            save(model,optimizer,output_model) #保存模型

            # 将训练过程中 最好的表现保存下来
            with open(write_best_score_path,'w+') as file:
                string1 = tag+ "\n" + "F1:" + str(best_f1)+"\n"+"F1_Macro:"+  str(best_score_F1_Macro) + "\n"+"F1_Micro:"+  str(best_score_F1_Micro)+ "\n"+"accuracy:"+  str(best_acc)+ "\n"+"precision:"+  str(best_pre)+ "\n"+"recall:"+  str(best_recall)
                file.write(string1)
                file.write("\n\n")
                string2 = "dev"+ "\n" +"F1:"+  str(temp_dev_f1)+"\n"+ "F1_Macro:"+  str(temp_dev_f1_macro) + "\n"+"F1_Micro:"+  str(temp_dev_f1_micro) + "\n"+"accuracy:"+  str(temp_dev_acc)+ "\n"+"precision:"+  str(temp_dev_pre)+ "\n"+"recall:"+  str(temp_dev_recall)
                file.write(string2)

    return best_f1

def testing(args, model ,features):
    if os.path.exists(output_model):
        print("=================loading_model_for_testing===================")
        model.load_state_dict(torch.load(output_model))

    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []
    for batch in dataloader:
        model.eval() #将模型设置为评估模式

        # batch {[input_id] [attention_mask] labels}  {[b*s] [b*s] }
        batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()  #从batch数据中取出lable 转为列表格式 再转存到keys中

        batch['labels'] = None
        with torch.no_grad():
            logits = model(args,**batch)[0]
            # preds += np.argmax(logits.cpu().numpy(), axis=-Data_set).flatten().tolist()  # 将模型的预测结果 转存到preds上
            preds+= np.argmax(logits.cpu().numpy(), axis=2).flatten().tolist()

    preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))

    preds = [ID_TO_LABEL[pred] for pred in preds]
    keys = [ID_TO_LABEL[key] for key in keys] #将index转为key形式

    # metrics
    f1_ = seqeval.metrics.f1_score([keys], [preds])
    F1_Macro = seqeval.metrics.f1_score([keys], [preds],average='macro')
    F1_Micro = seqeval.metrics.f1_score([keys], [preds],average='micro')
    acc=seqeval.metrics.accuracy_score([keys], [preds])
    precision=seqeval.metrics.precision_score([keys], [preds])
    recall=seqeval.metrics.recall_score([keys], [preds])

    output = {"F1": f1_,"F1_Macro": F1_Macro,"F1_Micro": F1_Micro,"accuracy": acc, "precision": precision, "recall": recall}
    print("test_result ",output)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #设备
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    #模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path[0]) #分词器

    model = BertSoftmax(args,args.model_name_or_path[0]) #模型
    model.to(args.device)

    train_file = args.train_file  # 数据位置
    dev_file = args.dev_file

    test_file = args.test_file

    # 将数据通过分词器转为特征
    if args.use_manual_template:  # 读手工模板数据
        train_features = read_manual_emplate_data(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_manual_emplate_data(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_test_data(test_file, tokenizer, max_seq_length=args.max_seq_length)
        re_test_features = read_test_data(test_file, tokenizer, max_seq_length=args.max_seq_length)

    elif args.soft_prompt:
        from Code.read_data.soft_prompt_read_data import read_data
        train_features = read_data(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_data(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_data(test_file, tokenizer, max_seq_length=args.max_seq_length)
        re_test_features = read_data(test_file, tokenizer, max_seq_length=args.max_seq_length)

    else:
        train_features = read_ner_data(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_ner_data(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_ner_data(test_file, tokenizer, max_seq_length=args.max_seq_length)
        re_test_features = read_ner_data(test_file, tokenizer, max_seq_length=args.max_seq_length)

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )

    best_f1_value = train(args, model, train_features, benchmarks) #调用train函数
    testing(args, model, re_test_features)
    return best_f1_value

if __name__ == "__main__":
    from Code.run_ner.MRC_Pointer_Set_Config import set_bert_softmax_config
    args_=set_bert_softmax_config()

    # args_=sys.argv[Data_set:]
    best_score_F1_Macro = 0  # 初始化 最好的得分为0
    best_score_F1_Micro = 0  #
    best_f1=0
    best_acc=0
    best_pre=0
    best_recall=0
    temp_dev_f1_macro=0
    temp_dev_f1_micro=0
    temp_dev_f1=0
    temp_dev_acc=0
    temp_dev_pre=0
    temp_dev_recall=0
    path=args_.loss_type+"_"+args_.model_dataset+"_"+str(args_.seed)
    mkpath = os.path.join(args_.output_model_root_path, path)
    mkdir(mkpath)     # 创建存储模型的文件夹

    output_model = os.path.join(mkpath, "model.pth")
    write_best_score_path = os.path.join(mkpath, "best_score.txt")
    write_training_info = os.path.join(mkpath, "training_info.txt")
    print(output_model)
    print(write_best_score_path)
    print(write_training_info)

    main(args_)












