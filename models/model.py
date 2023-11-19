import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
import numpy as np
from scipy import stats
import scipy
from Code.models.Wasserstein import SinkhornDistance
from Code.models.losses.focal_loss import FocalLoss
from Code.models.losses.label_smoothing import LabelSmoothingCrossEntropy
from Code.models.losses.diceLoss import DiceLoss
from Code.models.layers.crf import CRF

multi_class_tag = {"0":0,"B":1,"BI":2,"BII":3,"BIII":4,"BIIII":5,"BIIIII":6,"BIIIIII":7,"BIIIIIII":8}


def distance_(p, q ,args):
    args = args
    if args.dis_type == "KL":
        # # kl --------Data_set
        return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)
    elif args.dis_type == "BD":
        # # Bhattacharyya Distance --------2
        return  (-1)*((((p * q)**0.5).sum()).log())
    elif args.dis_type == "JS":
        # # Jensen-Shannon divergence --------3
        # M = (p + q) / 2
        # # return 0.5*(p*(p / M).log()).sum() + 0.5*(q*((q / M).log())).sum()
        return 1/2.0*((p*((p + 1e-5).log()-((p+q+2*(1e-5))/2.0).log())).sum()) + 1/2.0*((q*((q + 1e-5).log()-((p+q+2*(1e-5))/2.0).log())).sum())
    elif args.dis_type == "ED":
        # #Euclidean Distance --------4
        return (((p - q) ** 2).sum()) ** 0.5
    elif args.dis_type == "CE":
        # # crocess entropy --------5
        return (p * (q.log())).sum(-1)
    elif args.dis_type == "WD":
        # #wasserstein distance  --------6
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=30)
        dist, P, C = sinkhorn(p, q)
        return dist
    elif args.dis_type == "HD":
        # # Hellinger distance --------7
        return ((((p ** 0.5 - q ** 0.5) ** 2).sum()) ** 0.5) * (1.0 / (2 ** 0.5))


class NERModel(nn.Module):
    def __init__(self, args, model_name):
        super().__init__()
        self.args = args
        self.model_name_ = model_name

        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        self.model = AutoModel.from_pretrained(self.model_name_, )
        # self.init_weights()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_)  # 分词器
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_class)  # hidden_size 到 类别数
        if args.loss_type =="CE":
            self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)
        elif args.loss_type =="FL":
            self.loss_fnt = FocalLoss(ignore_index=-1)
            # self.loss_fnt = FocalLoss(gamma = 2, alpha = [Data_set.0] * 7)
        elif args.loss_type == "LS":
            self.loss_fnt = LabelSmoothingCrossEntropy(ignore_index=-1)
        elif args.loss_type == "DL":
            self.loss_fnt = DiceLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # print(input_ids.size(),attention_mask.size())
        h, *_ = self.model(input_ids, attention_mask, return_dict=False)  # 将input_id和att_mask 输入模型 得到隐状态
        h = self.dropout(h)  # 对隐状态 drop_out  [batch_size  sequence_lenth  hidden_size]

        c = self.args.num_class
        logits = self.classifier(h)  # [batch_size  sequence_lenth  class_num]

        logits = logits.view(-1, c)  # (batch_size*sequence_lenth  class_num) 隐状态h维度------>c维度

        # tensor(batch_size * sequence_lenth  class_num)
        outputs = (logits,)

        if labels is not None:  # lable [batch_size  sequence_lenth]
            labels = labels.view(-1)
            # xx=logits[0:len(labels)]
            loss = self.loss_fnt(logits[0:len(labels)], labels)  # 真实标签与 预测标签 之间取交叉熵
            # loss = self.loss_fnt(logits, labels)  # 真实标签与 预测标签 之间取交叉熵

            outputs = (loss,) + outputs  # 对batch 数据中的 损失相加
        return outputs

class CrfForNer(nn.Module):
    def __init__(self, args, model_name):
        super().__init__()
        self.args = args
        self.model_name_ = model_name

        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        self.model = AutoModel.from_pretrained(self.model_name_, )
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_class)
        self.crf = CRF(num_tags=args.num_class, batch_first=True)
        if args.loss_type == "CE":
            self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)
        elif args.loss_type == "FL":
            self.loss_fnt = FocalLoss(ignore_index=-1)
            # self.loss_fnt = FocalLoss(gamma = 2, alpha = [Data_set.0] * 7)
        elif args.loss_type == "LS":
            self.loss_fnt = LabelSmoothingCrossEntropy(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask, return_dict=False)
        h, *_ = self.model(input_ids, attention_mask, return_dict=False)
        sequence_output = h
        sequence_output = self.dropout(sequence_output)

        c = self.args.num_class
        logits = self.classifier(sequence_output)
        # logits = logits.view(-Data_set, c)
        # outputs = (logits,)

        # if labels is not None:
        #     loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
        #     logits = logits.view(-Data_set, c)
        #     outputs = (logits,)
        #
        #     outputs = (-Data_set * loss,) + outputs
        # logits = logits.view(-Data_set, c)
        # return outputs  # (loss), scores

        if labels is not None:
            crf_loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            # crf_loss = (-Data_set * crf_loss,)
            labels = labels.view(-1)
            logits = logits.view(-1, c)
            loss1 = self.loss_fnt(logits[0:len(labels)], labels)  # 真实标签与 预测标签 之间取交叉熵
            # loss1 = (loss1,)
            loss = crf_loss + loss1
            outputs = (logits,)

            outputs = (-1 * loss,) + outputs
        logits = logits.view(-1, c)
        return outputs  # (loss), scores


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class SpanForNer(nn.Module):
    def __init__(self, args, model_name):
        super().__init__()
        self.args = args
        self.model_name_ = model_name

        self.soft_label = True
        self.num_labels = 3
        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        self.loss_type = "CE"
        self.bert = AutoModel.from_pretrained(self.model_name_, )
        self.dropout = nn.Dropout(args.dropout_prob)

        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

    def forward(self,input_ids, attention_mask,start_positions=None,end_positions=None):
    # def forward(self,args, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None):

        outputs = self.bert(input_ids, attention_mask, return_dict=False)
        sequence_output = outputs[0]
        #[batch  seq_len  hidden_size]
        sequence_output = self.dropout(sequence_output)

        # [batch  seq_len  num_class]
        start_logits = self.start_fc(sequence_output)

        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)

                # [batch  seq_len  num_class]
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                #将label_logits放到设备上
                label_logits = label_logits.to(input_ids.device)
                # [batch  seq_len  num_class]
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)

            else: #soft_label为false
                label_logits = start_positions.unsqueeze(2).float()

        else: #start_positions 为 None 或 self.training为false:
            label_logits = F.softmax(start_logits, -1)

            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        # [batch  seq_len  num_class]
        end_logits = self.end_fc(sequence_output, label_logits)

        # (tensor[batch  seq_len  num_class] tensor[batch  seq_len  num_class])
        outputs = (start_logits, end_logits,) + outputs[2:]

        # start_positions、end_positions--[batch_size seq_len]
        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['CE']
            if self.loss_type =='CE':
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss()
            # else:
            #     loss_fct = CrossEntropyLoss()

            # [batch*seq_len  num_class]
            start_logits = start_logits.view(-1, self.num_labels)
            # [batch*seq_len  num_class]
            end_logits = end_logits.view(-1, self.num_labels)

            # [batch*seq_len]
            active_loss = attention_mask.view(-1) == 1
            # [batch*seq_len  num_class]


            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)

            total_loss = (start_loss + end_loss) / 2

            # {tensor[loss]  tensor[batch  seq_len  num_class]  tensor[batch  seq_len  num_class] }
            outputs = (total_loss,) + outputs
        return outputs

class Create_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        # self.device = [i % args.n_gpu for i in range(args.n_model)] #设备数
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))]  # 设备数
        # self.loss_fnt = nn.CrossEntropyLoss()

        # # for i in range(args.n_model):
        model_name_list = args.model_name_or_path
        for i in range(len(args.model_name_or_path)):
            model = NERModel(args, model_name_list[i])
            # model = CrfForNer(args, model_name_list[i])
            # model = SpanForNer(args, model_name_list[i])

            model.to(self.device[i])
            self.models.append(model)

    def forward(self,args, input_ids, attention_mask, labels=None):
        args_=args
        if labels is None:  # 标签为空--test
            return self.models[0](input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )
        else:
            num_models = len(self.models)
            # print("num_models",num_models)
            # print(self.models)
            outputs = []
            for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]) if labels is not None else None,
                )  # [len,num_class]

                output = tuple([o.to(0) for o in output])
                outputs.append(output)  # outputs里存放了 M个模型的输出

           # {[loss] [batch_size*hidensize, num_class]}
            model_output = outputs[0]

            loss = sum([output[0] for output in outputs]) / num_models
            loss_ = [output[0] for output in outputs]
            # print(loss_)

            logits = [output[1] for output in outputs] #logit为一个列表 里面包括模型数量个Tensor  每个Tensor为[batch_size*hiddemsize num_class]

            if args_.q_type =="reg":
                # 对logits取softmax  [[batch_size*hiddemsize num_class],[batch_size*hiddemsize num_class]]
                probs = [F.softmax(logit, dim=-1) for logit in logits]

                stack_prob = torch.stack(probs, dim=0) # [2 batch_size*hiddemsize num_class]
                avg_prob = stack_prob.mean(0) #每一列的平均值 [batch_size*hiddemsize num_class]

                # logits[0]  表示的是batch*seq_len 因为NER是token分类任务 要预测每个token的类别  与之相对于的label自然就有batch*seq_len个
                #将label 展开成一个[batch * seq_len]
                mask = (labels.view(-1) != -1).to(logits[0]) #[batch * seq_len== token label数]

                # prob为每个model 输出的概率分布    reg_loss为每个模型的预测 与平均预测 之间的距离-----的平均
                reg_loss = sum([distance_(avg_prob, prob ,args_) * mask for prob in probs]) / num_models

               # reg_loss = sum([distance_(avg_prob, prob ,distance_index) * mask for prob in probs]) / num_models
                reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                loss = loss + self.args.alpha_t * reg_loss # 联合训练的总损失

            elif args_.q_type =="mean":
                # 对logits取softmax  [[batch_size*hiddemsize num_class],[batch_size*hiddemsize num_class]]
                probs = [F.softmax(logit, dim=-1) for logit in logits]

                stack_prob = torch.stack(probs, dim=0) # [2 batch_size*hiddemsize num_class]
                avg_prob = stack_prob.mean(0) #每一列的平均值 [batch_size*hiddemsize num_class]

                # logits[0]  表示的是batch*seq_len 因为NER是token分类任务 要预测每个token的类别  与之相对于的label自然就有batch*seq_len个
                #将label 展开成一个[batch * seq_len]
                mask = (labels.view(-1) != -1).to(logits[0]) #[batch * seq_len== token label数]

                # prob为每个model 输出的概率分布    reg_loss为每个模型的预测 与平均预测 之间的距离-----的平均
                # reg_loss = sum([distance_(avg_prob, prob ,args_) * mask for prob in probs]) / num_models

                criteria = nn.CrossEntropyLoss(ignore_index=-1)
                labels_ = labels.view(-1)
                loss1 = criteria(avg_prob[0:len(labels_)], labels_)  # 真实标签与 预测标签 之间取交叉熵

                # reg_loss = sum([distance_(avg_prob, prob ,distance_index) * mask for prob in probs]) / num_models
                # reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
                # loss = loss + self.args.alpha_t * reg_loss + loss1 # 联合训练的总损失
                loss = loss + loss1 # 联合训练的总损失


            elif args_.q_type=="weight":
                probs = [F.softmax(logit, dim=-1) for logit in logits]
                loss_sum = sum([output[0] for output in outputs]).tolist()
                loss_weight_list = []
                for l in loss_:
                    l=l.tolist()
                    # loss_weight_list.append(l)
                    loss_weight_list.append((loss_sum-l+1e-8) / (loss_sum+1e-8))

                for i in range(len(probs)):
                    probs[i]=loss_weight_list[i]*probs[i]
                    # print(pro)

                stack_prob = torch.stack(probs, dim=0)
                weight_prob = stack_prob.sum(0)  #[batch_size * seq_len  num_class]

                mask = (labels.view(-1) != -1).to(logits[0])  # [batch * seq_len== token label数]
                # reg_loss = sum([distance_(weight_prob, prob,args_) * mask for prob in probs]) / num_models
                # reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                criteria = nn.CrossEntropyLoss(ignore_index=-1)
                labels_ = labels.view(-1)
                loss1 = criteria(weight_prob[0:len(labels_)], labels_)  # 真实标签与 预测标签 之间取交叉熵

                # loss = loss + self.args.alpha_t * reg_loss + loss1 # 联合训练的总损失
                loss = loss + loss1 # 联合训练的总损失

            # model_output[Data_set:]----(tensor[batch * seq_len])   (loss,)----(tensor[Data_set])
            # model_output -----(tensor[Data_set] , tensor[batch * seq_len])
            model_output_ = (loss,) + model_output[1:]

        return model_output_
