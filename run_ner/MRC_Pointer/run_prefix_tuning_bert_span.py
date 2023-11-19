import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from Code.models.learning_rate_scheduler import get_linear_schedule_with_warmup
from Code.read_data.model_input_feture import set_seed, collate_fn_span
from Code.read_data.read_file import read_ner_data, read_ner_data_span
from Code.read_data.save import save, mkdir
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
from Code.models.losses.diceLoss import DiceLoss
from Code.models.losses.focal_loss import FocalLoss
from Code.models.losses.label_smoothing import LabelSmoothingCrossEntropy
from Code.models.losses.ner_span_metrics import SpanEntityScore
from Code.models.adversarial import FGM
from Code.models.start_end_prefix import PoolerStartLogits, PoolerEndLogits, PrefixEncoder, bert_extract_item
from Code.run_ner.MRC_Pointer_Set_Config import SPAN_LABEL_TO_ID

SPAN_ID_TO_LABEL = {value: key for key, value in SPAN_LABEL_TO_ID.items()}


class BertSpan_prefix_tuning(nn.Module):
    def __init__(self, args, model_name):
        super().__init__()
        self.args = args
        self.model_name_ = model_name
        self.num_labels = args.num_class

        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        # config.prefix_projection = args.prefix_projection
        config.prefix_projection = False
        config.soft_label = True
        self.soft_label = config.soft_label

        self.model = AutoModel.from_pretrained(self.model_name_, )
        self.dropout = nn.Dropout(args.dropout_prob)
        # self.self_attention = MultiHeadedAttention(head_count=head_count, model_dim=size, dropout=dropout)
        # self.layer_norm = LayerNorm(size)

        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

        self.loss_type = args.loss_type

        # config.pre_seq_len = args.prefix_seq_len
        config.pre_seq_len = 11
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        # self.n_layer = 12
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.model.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        # 冻结BERT模型的所有参数而只训练这些prompts
        # for param in self.model.parameters():
        #     param.requires_grad=False

    def get_prompt(self, batch_size):
        # # [0-13]的tensor ([[0, Data_set, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
        # prefix_token0 = self.prefix_tokens.unsqueeze(0)
        # # [ batch  prefix_len ]
        # prefix_token1 = prefix_token0.expand(batch_size, -Data_set)
        # prefix_tokens = prefix_token1.to(self.model.device)  # [ batch  prefix_len ]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        # [batch_size  prefix_len  12*2*768]
        past_key_values = self.prefix_encoder(prefix_tokens)

        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values = self.dropout(past_key_values)

        # # [24 56 12 14 64]
        # past_key_value = past_key_values.permute([2, 0, 3, Data_set, 4])
        # # [2 56 12 14 64]  #第一个维度24按照2划分 分成了12个 2的
        # past_key_values = past_key_value.split(2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, args, input_ids, attention_mask, start_positions=None, end_positions=None):
        args = args
        batch_size = input_ids.shape[0]

        # （12个tensor） 每个tensor为[2  batch  num_layer  prefix_len  n_embd]
        past_key_values = self.get_prompt(batch_size=batch_size)

        # [batch  prefix_len] [Data_set 15]
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)

        # attention_mask[batch seq_len] [56 92]------>[56 106]
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                             return_dict=False)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        c = self.args.num_class

        attention_mask = attention_mask[:, self.pre_seq_len:].contiguous()

        start_logits = self.start_fc(sequence_output)  # 将序列隐状态给预测开始的分类器
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)

                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)  # [b s num_class]
                label_logits.zero_()

                label_logits = label_logits.to(input_ids.device)  # 放到GPU上
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)

            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        # attention_mask = attention_mask[:, self.pre_seq_len:].contiguous() #连续值

        if start_positions is not None and end_positions is not None:
            if args.loss_type == "CE":
                self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)
            elif args.loss_type == "FL":
                self.loss_fnt = FocalLoss(ignore_index=-1)
                # self.loss_fnt = FocalLoss(gamma = 2, alpha = [Data_set.0] * 7)
            elif args.loss_type == "LS":
                self.loss_fnt = LabelSmoothingCrossEntropy(ignore_index=-1)
            elif args.loss_type == "DL":
                self.loss_fnt = DiceLoss()

            ########
            start_logits = start_logits.view(-1, self.num_labels)  # [b*s num_class]
            end_logits = end_logits.view(-1, self.num_labels)  # [b*s num_class]
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1

                # active_start_logits = start_logits[active_loss] #b*s c
                # active_end_logits = end_logits[active_loss]  #b*s c
                # active_start_labels = start_positions.view(-Data_set)[active_loss]  #b*s
                # active_end_labels = end_positions.view(-Data_set)[active_loss] #b*s

                active_start_logits = start_logits.view(-1, c)  # b*s c
                active_end_logits = end_logits.view(-1, c)
                active_start_labels = torch.where(
                    active_loss, start_positions.view(-1),
                    torch.tensor(self.loss_fnt.ignore_index).type_as(start_positions)
                )
                active_end_labels = torch.where(
                    active_loss, end_positions.view(-1), torch.tensor(self.loss_fnt.ignore_index).type_as(end_positions)
                )

                start_loss = self.loss_fnt(active_start_logits, active_start_labels)
                end_loss = self.loss_fnt(active_end_logits, active_end_labels)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs
            ########
        return outputs  # loss , (start_logits, end_logits,) , outputs[2:]


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_span,
                                  drop_last=True)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()

    if args.do_adv:  # 对抗
        fgm = FGM(model, emb_name='word_embeddings', epsilon=1.0)

    num_steps = 0
    best_f1 = 0
    for epoch in range(int(args.num_train_epochs)):
        current_epoch = epoch + 1
        print("\n===================epoch====================   ", current_epoch)

        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:  # 还在热身warmup阶段
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha  # 下一阶段 引入一致性损失

            # batch = {key: value.to(args.device) for key, value in batch.items() if key != 'subjects'}  #将batch数据放到设备上
            batch = {key: value.to(args.device) for key, value in batch.items() if key != 'labels'}  # 将batch数据放到设备上
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "start_positions": batch["start_ids"], "end_positions": batch["end_ids"]}
            with autocast():
                outputs = model(args, **inputs)  # 将一批次数据输入模型中，得到输出

            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if args.do_adv:
                fgm.attack()
                loss_adv = model(args, **inputs)[0]
                loss_adv.backward()
                fgm.restore()

            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
            if step == len(train_dataloader) - 1:  # 训练已达最后一步
                for tag, features in benchmarks:
                    evaluate(args, model, optimizer, features, current_epoch, tag=tag)


def evaluate(args, model, optimizer, features, current_epoch, tag="dev_or_test"):
    # 保存训练信息
    with open(write_training_info, 'a+') as file1:
        str0 = "epoch:" + str(current_epoch)
        file1.write("\n")
        file1.write(str0)
        file1.write("\n")

    metric = SpanEntityScore(SPAN_ID_TO_LABEL)
    true_span_label, pred_span_to_label = [], []
    test_len = []
    eval_loss = 0.0
    nb_eval_steps = 0
    for step, f in enumerate(features):
        input_ids = torch.tensor([f['input_ids']], dtype=torch.long).to(args.device)
        input_ids_list = f['input_ids']
        test_len += input_ids_list

        start_ids = torch.tensor([f['start_ids']], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f['end_ids']], dtype=torch.long).to(args.device)
        subjects = f['subjects']
        input_mask = torch.tensor([f['input_mask']], dtype=torch.long).to(args.device)

        true_sp_label = f['labels']  # 真实标签
        true_sp_label0 = [la.replace("B-Disease", "Disease") for la in true_sp_label]
        true_sp_label1 = [la.replace("I-Disease", "Disease") for la in true_sp_label0]
        true_span_label += true_sp_label1

        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "start_positions": start_ids,
                      "end_positions": end_ids}

            logits = model(args, **inputs)

        tmp_eval_loss, start_logits, end_logits = logits[:3]
        # preds_span = bert_extract_item(start_logits, end_logits)
        # print("preds_span",preds_span)

        preds_span = bert_extract_item(start_logits, end_logits)
        true_span = subjects
        metric.update(true_subject=true_span, pred_subject=preds_span)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # pred_ = process_span(preds_span,len(input_ids_list))
        # print("pred_sp_label:",pred_)
        # print("true_sp_label:",true_sp_label1)
        # pred_span_to_label+=pred_
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    # return results

    model.zero_grad()
    # preds = pred_span_to_label
    # keys = true_span_label

    # metrics
    f1_ = results['f1']
    precision_ = results['precision']
    recall_ = results['recall']

    if tag == 'dev':
        global dev_f1_, dev_precision_, dev_recall_
        dev_f1_ = f1_
        dev_precision_ = precision_
        dev_recall_ = recall_
        print('\n')

    print(tag)
    output = {"F1": f1_, "presion": precision_, "recall": recall_}
    print(output)

    # 将训练过程中 训练信息保存下来
    with open(write_training_info, 'a+') as file2:
        str1 = tag + "\t" + "F1:" + str(f1_) + "\t" + "presion:" + str(precision_) + "\t" + "recall:" + str(recall_)
        file2.write(str1)
        file2.write("\n")

    if tag == 'test':
        global best_f1, best_pre, best_recall
        if (best_f1 < f1_):
            best_f1 = f1_
            best_presion = precision_
            best_recall = recall_
            save(model, optimizer, output_model)  # 保存模型

            # 将训练过程中 最好的表现保存下来
            with open(write_best_score_path, 'w+') as file:
                string1 = tag + "\n" + "F1:" + str(best_f1) + "\n" + "best_presion:" + str(
                    best_presion) + "\n" + "recall:" + str(best_recall)
                file.write(string1)
                file.write("\n\n")
                string2 = "dev" + "\n" + "F1:" + str(dev_f1_) + "\n" + "best_presion:" + str(
                    dev_precision_) + "\n" + "recall:" + str(dev_recall_)
                file.write(string2)


def testing(args, model, features):
    if os.path.exists(output_model):
        print("=================loading_model_for_testing===================")
        model.load_state_dict(torch.load(output_model))

    metric = SpanEntityScore(SPAN_ID_TO_LABEL)
    test_len = []
    for step, f in enumerate(features):
        input_ids = torch.tensor([f['input_ids']], dtype=torch.long).to(args.device)
        input_ids_list = f['input_ids']
        test_len += input_ids_list

        start_ids = torch.tensor([f['start_ids']], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f['end_ids']], dtype=torch.long).to(args.device)
        subjects = f['subjects']
        input_mask = torch.tensor([f['input_mask']], dtype=torch.long).to(args.device)
        # token_label = torch.tensor([f['label_token']], dtype=torch.long).to(args.device)

        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "start_positions": start_ids,
                      "end_positions": end_ids, }
            # inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_label":token_label, "start_positions": start_ids, "end_positions": end_ids , }
            logits = model(args, **inputs)

        tmp_eval_loss, start_logits, end_logits = logits[:3]

        preds_span = bert_extract_item(start_logits, end_logits)
        true_span = subjects
        metric.update(true_subject=true_span, pred_subject=preds_span)

    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}

    f1_ = results['f1']
    precision_ = results['precision']
    recall_ = results['recall']

    output = {"F1": f1_, "presion": precision_, "recall": recall_}
    print("testing:", output)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    # 模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path[0])  # 分词器

    model = BertSpan_prefix_tuning(args, args.model_name_or_path[0])  # 模型
    model.to(args.device)

    train_file = args.train_file  # 数据位置
    dev_file = args.dev_file
    test_file = args.test_file

    # 将数据通过分词器转为特征
    if args.span_ner:
        train_features = read_ner_data_span(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_ner_data_span(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_ner_data_span(test_file, tokenizer, max_seq_length=args.max_seq_length)

    else:
        train_features = read_ner_data(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_ner_data(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_ner_data(test_file, tokenizer, max_seq_length=args.max_seq_length)

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )

    train(args, model, train_features, benchmarks)  # 调用train函数
    testing(args, model, test_features)


if __name__ == "__main__":
    from Code.run_ner.MRC_Pointer_Set_Config import set_bert_span_prefix_tuning_config

    args_ = set_bert_span_prefix_tuning_config()

    dev_f1_ = 0
    dev_precision_ = 0
    dev_recall_ = 0
    best_f1 = 0
    best_pre = 0
    best_recall = 0
    # path=str(i)+args_.loss_type+"_"+args_.model_dataset+"_"+str(args_.seed)+"_"+str(args_.prefix_seq_len)
    path = args_.loss_type + "_" + args_.model_dataset + "_" + str(args_.seed) + "_" + str(args_.prefix_seq_len)
    mkpath = os.path.join(args_.output_model_root_path, path)
    mkdir(mkpath)  # 创建存储模型的文件夹

    output_model = os.path.join(mkpath, "model.pth")
    write_best_score_path = os.path.join(mkpath, "best_score.txt")
    write_training_info = os.path.join(mkpath, "training_info.txt")
    print(output_model)
    print(write_best_score_path)
    print(write_training_info)
    main(args_)
    # i+=Data_set
