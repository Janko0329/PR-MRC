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
from Code.models.layers.soft_embedding import SoftEmbedding


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
        n_tokens = args.soft_prompt_token
        initialize_from_vocab = True

        config = AutoConfig.from_pretrained(self.model_name_, num_labels=args.num_class)
        self.model = AutoModel.from_pretrained(self.model_name_, )
        self.model.get_input_embeddings()

        s_wte = SoftEmbedding(self.model.get_input_embeddings(),
                              n_tokens=n_tokens,
                              initialize_from_vocab=initialize_from_vocab)
        self.model.set_input_embeddings(s_wte)


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
        h, *_ = self.model(input_ids, attention_mask, return_dict=False)
        # h = h[0][:,20:-2,:]
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

class Create_soft_prompt_model(nn.Module):
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
            # model = NERModel(args.model_name_or_path[i])
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

            outputs = []
            for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]) if labels is not None else None,
                )

                output = tuple([o.to(0) for o in output])
                outputs.append(output)  # outputs里存放了 M个模型的输出

           # {[loss] [batch_size*hidensize, num_class]}
            model_output = outputs[0]

            loss = sum([output[0] for output in outputs]) / num_models
            loss_ = [output[0] for output in outputs]

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

            # model_output[Data_set:]----(tensor[batch * seq_len])   (loss,)----(tensor[Data_set])
            # model_output -----(tensor[Data_set] , tensor[batch * seq_len])
            model_output_ = (loss,) + model_output[1:]

        return model_output_
