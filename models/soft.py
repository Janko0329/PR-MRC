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
    elif args.dis_type == "WD":
        # #wasserstein distance  --------6
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=30)
        dist, P, C = sinkhorn(p, q)
        return dist
    elif args.dis_type == "HD":
        # # Hellinger distance --------7
        return ((((p ** 0.5 - q ** 0.5) ** 2).sum()) ** 0.5) * (1.0 / (2 ** 0.5))
    elif args.dis_type == "ED":
        # #Euclidean Distance --------4
        return (((p - q) ** 2).sum()) ** 0.5
    elif args.dis_type == "CE":
        # # crocess entropy --------5
        return (p * (q.log())).sum(-1)


from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM
class MyModel(BertForMaskedLM):
    # def __init__(self, args, model_name):
    #     super().__init__()
    #     self.args = args
    #     self.model_name_ = model_name
    def __init__(self,config):
        super().__init__(config)

        prp_len = 14
        self.model_name_ = 'bert-base-cased'
        config = AutoConfig.from_pretrained(self.model_name_, num_labels=3)

        self.bert = AutoModel.from_pretrained(self.model_name_, )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

        # def get_input_embeddings(self):
        #    return self.embeddings.word_embeddings

        self.dim = 384
        self.emb = nn.Embedding(prp_len + 1, self.dim)
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True)
        self.b_emb = self.get_input_embeddings()
        # self.b_emb = self.bi_lstm
        self.line1 = nn.Linear(768, 768)
        self.line2 = nn.Linear(768, 768)
        self.line3 = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(
            self,
            prp_len=2,
            input_ids=None,  # [CLS] e(p) e(p) [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        device='cuda'
        p = self.emb(torch.LongTensor([range(1, prp_len + 1)] * input_ids.shape[0]).to(device))  # 若用GPU则要注意将数据导入cuda
        p = self.bi_lstm(p)[0]
        p = self.relu(self.line1(p))
        p = self.relu(self.line2(p))
        p = self.relu(self.line3(p))
        inputs_embeds = self.b_emb(input_ids)
        inputs_embeds[:, 1:prp_len + 1, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        h = outputs[0]
        h = self.dropout(h)
        c = 3
        logits = self.classifier(h)
        # prediction_scores = self.cls(sequence_output)
        logits = logits.view(-1, c)
        outputs = (logits,)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)  # -100 index = padding token
            # masked_lm_loss = loss_fct(prediction_scores.view(-Data_set, self.config.vocab_size), labels.view(-Data_set))
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
        # if not return_dict:
        #     output = (prediction_scores,) + outputs[2:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # return MaskedLMOutput(
        #     loss=masked_lm_loss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


class Create_frefix_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        # self.device = [i % args.n_gpu for i in range(args.n_model)] #设备数
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))] #设备数
        self.loss_fnt = nn.CrossEntropyLoss()


        # # for i in range(args.n_model):
        model_name_list=args.model_name_or_path
        for i in range(len(args.model_name_or_path)):
            model = MyModel.from_pretrained(model_name_list[i])
            model.to(self.device[i])
            self.models.append(model)
        print(len(self.models),"  ",model_name_list)


    def forward(self,args, input_ids, attention_mask, labels=None):
        args_ = args
        if labels is None:  #标签为空--test
            return self.models[0](input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )
        else:
            num_models = len(self.models)

            outputs = []
            for i in range(num_models):  #将input_id att_mask label 输入模型   得到模型[i]的输出
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]) if labels is not None else None,
                )


                output = tuple([o.to(0) for o in output])
                outputs.append(output) #outputs里存放了 M个模型的输出

            model_output = outputs[0]

            loss = sum([output[0] for output in outputs]) / num_models
            loss_ = [output[0] for output in outputs]
            logits = [output[1] for output in outputs]

            if args_.q_type == "reg":
                # 对logits取softmax  [[batch_size*hiddemsize num_class],[batch_size*hiddemsize num_class]]
                probs = [F.softmax(logit, dim=-1) for logit in logits]

                stack_prob = torch.stack(probs, dim=0)  # [2 batch_size*hiddemsize num_class]
                avg_prob = stack_prob.mean(0)  # 每一列的平均值 [batch_size*hiddemsize num_class]

                # logits[0]  表示的是batch*seq_len 因为NER是token分类任务 要预测每个token的类别  与之相对于的label自然就有batch*seq_len个
                # 将label 展开成一个[batch * seq_len]
                mask = (labels.view(-1) != -1).to(logits[0])  # [batch * seq_len== token label数]

                # prob为每个model 输出的概率分布    reg_loss为每个模型的预测 与平均预测 之间的距离-----的平均
                reg_loss = sum([distance_(avg_prob, prob, args_) * mask for prob in probs]) / num_models

                criteria = nn.CrossEntropyLoss(ignore_index=-1)
                labels_ = labels.view(-1)
                loss1 = criteria(avg_prob[0:len(labels_)], labels_)  # 真实标签与 预测标签 之间取交叉熵

                # reg_loss = sum([distance_(avg_prob, prob ,distance_index) * mask for prob in probs]) / num_models
                reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                loss = loss + self.args.alpha_t * reg_loss  # 联合训练的总损失

            if args_.q_type == "mean":
                # 对logits取softmax  [[batch_size*hiddemsize num_class],[batch_size*hiddemsize num_class]]
                probs = [F.softmax(logit, dim=-1) for logit in logits]

                stack_prob = torch.stack(probs, dim=0)  # [2 batch_size*hiddemsize num_class]
                avg_prob = stack_prob.mean(0)  # 每一列的平均值 [batch_size*hiddemsize num_class]

                # logits[0]  表示的是batch*seq_len 因为NER是token分类任务 要预测每个token的类别  与之相对于的label自然就有batch*seq_len个
                # 将label 展开成一个[batch * seq_len]
                mask = (labels.view(-1) != -1).to(logits[0])  # [batch * seq_len== token label数]

                # prob为每个model 输出的概率分布    reg_loss为每个模型的预测 与平均预测 之间的距离-----的平均
                reg_loss = sum([distance_(avg_prob, prob, args_) * mask for prob in probs]) / num_models

                criteria = nn.CrossEntropyLoss(ignore_index=-1)
                labels_ = labels.view(-1)
                loss1 = criteria(avg_prob[0:len(labels_)], labels_)  # 真实标签与 预测标签 之间取交叉熵

                # reg_loss = sum([distance_(avg_prob, prob ,distance_index) * mask for prob in probs]) / num_models
                reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                loss = loss + self.args.alpha_t * reg_loss + loss1  # 联合训练的总损失


            elif args_.q_type == "weight":
                probs = [F.softmax(logit, dim=-1) for logit in logits]
                loss_sum = sum([output[0] for output in outputs]).tolist()
                loss_weight_list = []
                for l in loss_:
                    l = l.tolist()
                    # loss_weight_list.append(l)
                    loss_weight_list.append((loss_sum - l + 1e-8) / (loss_sum + 1e-8))

                for i in range(len(probs)):
                    probs[i] = loss_weight_list[i] * probs[i]
                    # print(pro)

                stack_prob = torch.stack(probs, dim=0)
                weight_prob = stack_prob.sum(0)  # [batch_size * seq_len  num_class]

                mask = (labels.view(-1) != -1).to(logits[0])  # [batch * seq_len== token label数]
                reg_loss = sum([distance_(weight_prob, prob, args_) * mask for prob in probs]) / num_models
                reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                criteria = nn.CrossEntropyLoss(ignore_index=-1)
                labels_ = labels.view(-1)
                loss1 = criteria(weight_prob[0:len(labels_)], labels_)  # 真实标签与 预测标签 之间取交叉熵

                loss = loss + self.args.alpha_t * reg_loss + loss1  # 联合训练的总损失

            # model_output[Data_set:]----(tensor[batch * seq_len])   (loss,)----(tensor[Data_set])
            # model_output -----(tensor[Data_set] , tensor[batch * seq_len])
            model_output = (loss,) + model_output[1:]

        return model_output

