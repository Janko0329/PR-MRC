
import math
import sys

sys.path.append("../")
from Code.read_data.utils import Preprocessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

class DataMaker(object): #数据
    def __init__(self, tokenizer, add_special_tokens = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        #填充成最大长度
        for sample in datas:
            input_id=sample["input_ids"]
            att_mask=sample["input_mask"]
            token_type_id=sample["input_ids"]
            label_token = sample["label_token"]

            input_ids = [input_id + [0] * (max_seq_len - len(input_id))]
            attention_mask = [att_mask + [0] * (max_seq_len - len(att_mask))]
            token_type_ids = [[0] * len(token_type_id) + [0] * (max_seq_len - len(token_type_id))]
            label_token = [label_token + [-1] * (max_seq_len - len(label_token))]

            labels = None
            if data_type != "test":
                # temp_str = sample["text"].split(" ")
                # ent2token_spans = self.preprocessor.get_ent2token_spans(
                #     # temp_str, sample["entity_list"]
                #     sample["text"], sample["entity_list"])
                ent2token_spans=sample["subjects"]

                # token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for label, start, end in ent2token_spans:
                    entity_label_id=str(label)
                    labels[ent2id[entity_label_id], start, end]=1
            # inputs["labels"] = labels

            input_ids = torch.tensor(input_ids).long()
            attention_mask = torch.tensor(attention_mask).long()
            token_type_ids = torch.tensor(token_type_ids).long()
            label_token = torch.tensor(label_token).long()
            if labels is not None:
                labels = torch.tensor(labels).long()

            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels ,label_token)

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train",):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []
        labels_token_list=[]

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1]) #sample[1] input_ids
            attention_mask_list.append(sample[2]) #sample[2] attention_mask
            token_type_ids_list.append(sample[3]) #sample[3] token_type_ids
            if data_type != "test":
                labels_list.append(sample[4]) #sample[4] label
            labels_token_list.append(sample[5])

        #生成batch数据
        batch_input_ids = torch.stack(input_ids_list, dim=1)
        batch_input_ids=batch_input_ids[0]
        batch_attention_mask = torch.stack(attention_mask_list, dim=1)
        batch_attention_mask=batch_attention_mask[0]
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=1)
        batch_token_type_ids=batch_token_type_ids[0]
        batch_labels = torch.stack(labels_list, dim=0) if data_type!="test" else None #labels
        batch_label_token = torch.stack(labels_token_list, dim=1)
        batch_label_token=batch_label_token[0]

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,batch_label_token

    def decode_ent(self, pred_matrix):
        pass

class MetricsCalculator(object): #评价指标
    def __init__(self):
        super().__init__()
    
    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)
    
    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum()+1)
    
    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        # for b, l, start, end in zip(*np.where(y_pred>0)):
        #     pred.append((b, l, start, end))
        # for b, l, start, end in zip(*np.where(y_true>0)):
        #     true.append((b, l, start, end))
        for _, l, start, end in zip(*np.where(y_pred>0)):
            pred.append((l, start, end))
        for _, l, start, end in zip(*np.where(y_true>0)):
            true.append((l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z +1e-8), X / (Y++1e-8), X / (Z++1e-8)
        return f1, precision, recall

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    '''Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

    # 第二种实现
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids

class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class GlobalPointer(nn.Module): #Globalpointer模型
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size #10
        self.inner_dim = inner_dim #64
        self.hidden_size = 768

        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2) #（768 1280）
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim): #位置编码
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1) #句子中token的位置编码 0至len-1

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  #0到output_dim/2
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices  #tensor(seq,indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) #[seq,indices,2]
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape)))) #[b,seq,indices,2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))  #[batch_size, seq_len, output_dim]
        embeddings = embeddings.to(self.device)
        return embeddings

        # input_ids, attention_mask, token_type_ids  --->[batch,seq]
    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0] #[b s 768]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state) #经过全连接 [b s 1280]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1) #在最后的维度上 按inner_dim * 2划分 为实体类型数 那么多的组
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2) #[b s 组数 inner_dim*2]

        #qk和qv分别为(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:] # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)

            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)

            #(batch_size, seq_len, ent_type_size, inner_dim)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape) #(batch_size, seq_len, ent_type_size, inner_dim)

            qw = qw * cos_pos + qw2 * sin_pos #(batch_size, seq_len, ent_type_size, inner_dim)

            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1) #(batch_size, seq_len, ent_type_size, inner_dim/2, 2)
            kw2 = kw2.reshape(kw.shape) #(batch_size, seq_len, ent_type_size, inner_dim)

            kw = kw * cos_pos + kw2 * sin_pos #(batch_size, seq_len, ent_type_size, inner_dim)

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask : (batch_size, ent_type_size, seq_len, seq_len)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h

        #logits: (batch_size, ent_type_size, seq_len, seq_len)
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        # mask：(batch_size, ent_type_size, seq_len, seq_len)
        mask = torch.tril(torch.ones_like(logits), -1)

        # logits: (batch_size, ent_type_size, seq_len, seq_len)
        logits = logits - mask * 1e12

        # (batch_size, ent_type_size, seq_len, seq_len)
        temp=logits/self.inner_dim**0.5 #logits除以 根号inner_dim

        return temp
        # return logits/self.inner_dim**0.5

class regularization_GlobalPointer(nn.Module): #Globalpointer模型
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size #10
        self.inner_dim = inner_dim #64
        self.hidden_size = 768

        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2) #（768 1280）
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim): #位置编码
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1) #句子中token的位置编码 0至len-1

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  #0到output_dim/2
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices  #tensor(seq,indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) #[seq,indices,2]
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape)))) #[b,seq,indices,2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))  #[batch_size, seq_len, output_dim]
        embeddings = embeddings.to(self.device)
        return embeddings

        # input_ids, attention_mask, token_type_ids  --->[batch,seq]
    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0] #[b s 768]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state) #经过全连接 [b s 1280]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1) #在最后的维度上 按inner_dim * 2划分 为实体类型数 那么多的组
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2) #[b s 组数 inner_dim*2]

        #qk和qv分别为(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:] # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)

            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)

            #(batch_size, seq_len, ent_type_size, inner_dim)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape) #(batch_size, seq_len, ent_type_size, inner_dim)

            qw = qw * cos_pos + qw2 * sin_pos #(batch_size, seq_len, ent_type_size, inner_dim)

            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1) #(batch_size, seq_len, ent_type_size, inner_dim/2, 2)
            kw2 = kw2.reshape(kw.shape) #(batch_size, seq_len, ent_type_size, inner_dim)

            kw = kw * cos_pos + kw2 * sin_pos #(batch_size, seq_len, ent_type_size, inner_dim)

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask : (batch_size, ent_type_size, seq_len, seq_len)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h

        #logits: (batch_size, ent_type_size, seq_len, seq_len)
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        # mask：(batch_size, ent_type_size, seq_len, seq_len)
        mask = torch.tril(torch.ones_like(logits), -1)

        # logits: (batch_size, ent_type_size, seq_len, seq_len)
        logits = logits - mask * 1e12

        # (batch_size, ent_type_size, seq_len, seq_len)
        temp_logits=logits/self.inner_dim**0.5 #logits除以 根号inner_dim
        out_result=(temp_logits,)
        loss = criterion(temp_logits, batch_labels) #与真实标签之间的 损失

        out_result=(loss,) + out_result
        return out_result

class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size

        self.heads = self.ent_type_size
        # self.hidden_size = encoder.config.hidden_size
        self.hidden_size = 768
        self.head_size = inner_dim
        self.RoPE = RoPE
        self.tril_mask = True
        self.RoPE = RoPE

        self.p_dense = nn.Linear(self.hidden_size, self.head_size * 2)
        self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2)

        def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):  # 位置编码
            position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # 句子中token的位置编码 0至len-1

            indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # 0到output_dim/2
            indices = torch.pow(10000, -2 * indices / output_dim)
            embeddings = position_ids * indices  # tensor(seq,indices)
            embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq,indices,2]
            embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [b,seq,indices,2]
            embeddings = torch.reshape(embeddings,
                                       (batch_size, seq_len, output_dim))  # [batch_size, seq_len, output_dim]
            embeddings = embeddings.to(self.device)
            return embeddings

            # input_ids, attention_mask, token_type_ids  --->[batch,seq]

        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(512, self.head_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        outputs0 = context_outputs[0]  # [b s 768]

        # atten=context_outputs["attentions"] #用各层的attention
        # all_hidden_state=context_outputs["hidden_states"][1:]
        # outputs1=torch.stack(all_hidden_state[-4:],dim=0).mean(0) #取最后4层的均值
        #
        # # temp_out=nn.MaxPool1d(outputs1)
        # #取倒数4层的hidden_state
        #
        # # atten=torch.tensor(atten)
        # # att=atten.view(1,1,12)
        # # for i in range(0,12):
        # #     outputs1=outputs1+atten[i]*all_hidden_state[i]

        batch_size = outputs0.size()[0]
        seq_len = outputs0.size()[1]

        sequence_output = self.p_dense(outputs0)  # [..., head_size*2]
        # sequence_output = self.p_dense(outputs1)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # if mask is not None:
        #     attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
        #     attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
        #     logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
        #     logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # if self.tril_mask:
        #     logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits

class regularization_EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size

        self.heads = self.ent_type_size
        # self.hidden_size = encoder.config.hidden_size
        self.hidden_size = 768
        self.head_size = inner_dim
        self.RoPE = RoPE
        self.tril_mask = True
        self.RoPE = RoPE

        self.p_dense = nn.Linear(self.hidden_size, self.head_size * 2)
        self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2)

        def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):  # 位置编码
            position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # 句子中token的位置编码 0至len-1

            indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # 0到output_dim/2
            indices = torch.pow(10000, -2 * indices / output_dim)
            embeddings = position_ids * indices  # tensor(seq,indices)
            embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq,indices,2]
            embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [b,seq,indices,2]
            embeddings = torch.reshape(embeddings,
                                       (batch_size, seq_len, output_dim))  # [batch_size, seq_len, output_dim]
            embeddings = embeddings.to(self.device)
            return embeddings

            # input_ids, attention_mask, token_type_ids  --->[batch,seq]

        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(512, self.head_size)

    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        outputs0 = context_outputs[0]  # [b s 768]

        atten=context_outputs["attentions"]
        all_hidden_state=context_outputs["hidden_states"][1:]
        outputs1=torch.stack(all_hidden_state[-4:],dim=0).mean(0) #取最后4层的均值

        # temp_out=nn.MaxPool1d(outputs1)
        #取倒数4层的hidden_state

        # atten=torch.tensor(atten)
        # att=atten.view(1,1,12)
        # for i in range(0,12):
        #     outputs1=outputs1+atten[i]*all_hidden_state[i]

        batch_size = outputs0.size()[0]
        seq_len = outputs0.size()[1]

        # sequence_output = self.p_dense(outputs0)  # [..., head_size*2]
        sequence_output = self.p_dense(outputs1)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # if mask is not None:
        #     attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
        #     attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
        #     logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
        #     logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # if self.tril_mask:
        #     logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        temp_logits = logits - mask * 1e12

        out_result = (temp_logits,)
        loss = criterion(temp_logits, batch_labels)  # 与真实标签之间的 损失

        out_result = (loss,) + out_result

        return out_result

# class EfficientGlobalPointer(nn.Module):
#     def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
#         super().__init__()
#         self.encoder = encoder
#         self.ent_type_size = ent_type_size  #
#
#         self.heads=self.ent_type_size
#         hidden_size = encoder.config.hidden_size
#         self.head_size = inner_dim
#         self.RoPE = RoPE
#         self.tril_mask = True
#         self.RoPE = RoPE
#
#         # self.p_dense = nn.Linear(hidden_size, self.head_size * 2,bias=True)
#         # self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2,bias=True)
#         self.p_dense = nn.Linear(hidden_size, self.head_size * 2)
#         self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2)
#
#         # if self.RoPE:
#         #     self.position_embedding = RoPEPositionEncoding(512, self.head_size)
#
#     def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):  # 位置编码
#         position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # 句子中token的位置编码 0至len-1
#
#         indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # 0到output_dim/2
#         indices = torch.pow(10000, -2 * indices / output_dim)
#         embeddings = position_ids * indices  # tensor(seq,indices)
#         embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq,indices,2]
#         embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [b,seq,indices,2]
#         embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))  # [batch_size, seq_len, output_dim]
#         embeddings = embeddings.to(self.device)
#         return embeddings
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         self.device = input_ids.device
#
#         context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
#         last_hidden_state = context_outputs[0]  # [b s 768]
#
#         batch_size = last_hidden_state.size()[0]
#         seq_len = last_hidden_state.size()[1]
#
#         sequence_output = self.p_dense(last_hidden_state)  #(b ,s , head_size * 2)
#         sequence_output = torch.split(sequence_output, self.head_size * 2, dim=-1)  # 在最后的维度上 按inner_dim * 2划分 为实体类型数 那么多的组
#         # sequence_output = torch.stack(sequence_output, dim=-2)  # outputs:(batch_size, seq_len, inner_dim*2)
#         sequence_output = torch.stack(sequence_output, dim=0)[0]  # outputs:(batch_size, seq_len, inner_dim*2)
#
#         qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [b,s, head_size]
#
#         # if self.RoPE: #qw km [b s head_num]
#         #     qw = self.position_embedding(qw)
#         #     kw = self.position_embedding(kw)
#
#         # ROPE编码
#         if self.RoPE:
#             # pos_emb:(batch_size, seq_len, inner_dim)
#             pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size)
#
#             # cos_pos,sin_pos: (batch_size, seq_len, inner_dim)
#             cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
#             cos_pos=cos_pos[:,:,0,:]
#             sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
#             sin_pos=sin_pos[:,:,0,:]
#
#             # (batch_size, seq_len, ent_type_size, inner_dim)
#             qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
#             qw2 = qw2.reshape(qw.shape)  # (batch_size, seq_len, ent_type_size, inner_dim)
#
#             qw = qw * cos_pos + qw2 * sin_pos  # (batch_size, seq_len, inner_dim)
#
#             kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]],
#                               -1)  # (batch_size, seq_len, inner_dim/2, 2)
#             kw2 = kw2.reshape(kw.shape)  # (batch_size, seq_len, inner_dim)
#
#             kw = kw * cos_pos + kw2 * sin_pos  # (batch_size, seq_len, ent_type_size, inner_dim)
#
#         #计算内积
#         logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
#         bias_input = self.q_dense(sequence_output)  # [..., heads*2]
#         bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, heads, seq_len, 2]
#         logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]
#
#         # 排除padding
#         pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
#         # mask=attention_mask
#         # if mask is not None:
#         #     attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
#         #     attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
#         #     logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf')) #[btz, head, seq, seq]
#         #     logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf')) #[btz, head, seq, seq]
#
#         logits = logits * pad_mask - (1 - pad_mask) * 1e12
#
#         # # 排除下三角
#         # if self.tril_mask:
#         #     logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12 #[btz, head, seq, seq]
#
#         mask = torch.tril(torch.ones_like(logits), -1)
#
#         logits = logits - mask * 1e12
#
#         # return logits / self.head_size ** 0.5
#         return logits


class PrefixEncoder(torch.nn.Module):
    r'''
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        config.prefix_hidden_size = 768

        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)

            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                # [batch, prefix_len, 12*2*768]
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            # [batch, prefix_len, 12*2*768]
            # self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)
            self.embedding = torch.nn.Embedding(config.pre_seq_len, 12 * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:  # two-layer MLP to encode the prefix
            # [batch, prefix_len, 768]
            prefix_tokens = self.embedding(prefix)
            # [batch, prefix_len, 12*2*768]
            past_key_values = self.trans(prefix_tokens)
        else:
            # [batch_size  prefix_len  12*2*768]
            past_key_values = self.embedding(prefix)
        return past_key_values

class Encoder_Bert_prefix(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model_name_ = model_name

        from transformers import AutoConfig,AutoModel
        config = AutoConfig.from_pretrained(self.model_name_,)

        from Code.run_ner.MRC_EG_G_Set_Config import bert_prefix_globalpointer_config
        args1 = bert_prefix_globalpointer_config()

        config.prefix_projection = args1.prefix_projection
        config.dropout_prob=args1.dropout_prob
        config.pre_seq_len=args1.pre_seq_len

        config.soft_label = True
        self.soft_label = config.soft_label

        config.output_hidden_states = True  # 输出allhidden
        config.output_attentions = True
        self.model = AutoModel.from_pretrained(self.model_name_, config=config)
        self.dropout = nn.Dropout(config.dropout_prob)
        # self.self_attention = MultiHeadedAttention(head_count=head_count, model_dim=size, dropout=dropout)
        # self.layer_norm = LayerNorm(size)

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

    def forward(self,input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        batch_size = input_ids.shape[0]

        # （12个tensor） 每个tensor为[2  batch  num_layer  prefix_len  n_embd]
        past_key_values = self.get_prompt(batch_size=batch_size)

        # [batch  prefix_len] [Data_set 15]
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)

        # attention_mask[batch seq_len] [56 92]------>[56 106]
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                             return_dict=True)
        return outputs

class Encoder_Bert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name_ = model_name

        from transformers import AutoConfig,AutoModel
        config = AutoConfig.from_pretrained(self.model_name_,)

        config.dropout_prob=0.1
        config.soft_label = True
        self.soft_label = config.soft_label
        config.output_hidden_states = True  # 输出allhidden
        config.output_attentions = True

        self.model = AutoModel.from_pretrained(self.model_name_, config=config)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self,input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        outputs = self.model(input_ids, attention_mask=attention_mask,return_dict=True)
        return outputs

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
    # elif args.dis_type == "WD":
    #     # #wasserstein distance  --------6
    #     sinkhorn = SinkhornDistance(eps=0.1, max_iter=30)
    #     dist, P, C = sinkhorn(p, q)
    #     return dist
    elif args.dis_type == "HD":
        # # Hellinger distance --------7
        return ((((p ** 0.5 - q ** 0.5) ** 2).sum()) ** 0.5) * (1.0 / (2 ** 0.5))
    elif args.dis_type == "ED":
        # #Euclidean Distance --------4
        return (((p - q) ** 2).sum()) ** 0.5
    elif args.dis_type == "CE":
        # # crocess entropy --------5
        return (p * (q.log())).sum(-1)

class create_bert_GlobalPointer_regularization(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))]  # 设备数

        for i in range(len(args.model_name_or_path)):
            encoder = Encoder_Bert(args.model_name_or_path[i])
            model = regularization_GlobalPointer(encoder,args.ent_type_size, 64)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion,batch_label_token):
        batch_size = input_ids.shape[0]

        num_models = len(self.models)
        outputs = []
        for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                token_type_ids=token_type_ids.to(self.device[i]),
                batch_labels=batch_labels.to(self.device[i]),
                criterion=criterion
            )

            output = tuple([o.to(0) for o in output])
            outputs.append(output)  # outputs里存放了 M个模型的输出

        loss_with_label = sum([output[0] for output in outputs]) / num_models
        model_output = outputs[0][1]  # 因为多个model的预测用kl散度进行靠近 所以返回第一个model的预测即可

        regularization_logits = [output[1] for output in outputs]  # model i

        ########################## #取mean_logits 作为软logits
        logits_stack_out = torch.stack(regularization_logits, dim=0)
        mean_logits_out = logits_stack_out.mean(0)
        mean_outputs_logits = (mean_logits_out,)
        ##########################

        from Code.run_ner.MRC_EG_G_Set_Config import bert_GlobalPointer_regularization_config
        args_ = bert_GlobalPointer_regularization_config()
        num_type = args_.ent_type_size

        import torch.nn.functional as F
        if args_.fit_type=="reg":
                # avg_logits.transpose(0, 1)
                # regularization_avg_logits = avg_logits.reshape(args_.ent_type_size,-1)  # [entity_type b s s]

                # logits_tran = [prob.transpose(0,1) for prob in probs1]
                # logits_ = [prob.reshape(args_.ent_type_size,-1) for prob in logits_tran]

                # token_input_label = (batch_label_token.view(-1) != -1).to(regularization_avg_logits)
                # reg_loss = sum([distance_(regularization_avg_logits, prob, args_) * token_input_label for prob in logits_]) / num_models

            avg_logits = mean_logits_out.reshape(-1)  #
            avg_logits = F.softmax(avg_logits,dim=0)  #
            logits_ = [prob.reshape(-1) for prob in regularization_logits]
            logits_ = [F.softmax(prob,dim=0) for prob in logits_]

            reg_loss = sum([distance_(avg_logits, prob, args_) for prob in logits_]) / num_models

            # reg_loss_sum = reg_loss.sum()
            # loss_out = loss_with_label + args_.alpha_t * reg_loss  # 联合训练的总损失
            loss_out = loss_with_label + reg_loss # 联合训练的总损失

        # model_output = (loss_out,) + mean_outputs_logits  #mean_logits 软标签
        models_output = (loss_out,) + (model_output,) #loss+kl + 第一个模块的标签

        return models_output

class create_bert_prefix_GlobalPointer_regularization(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))]  # 设备数

        for i in range(len(args.model_name_or_path)):
            encoder=Encoder_Bert_prefix(args.model_name_or_path[i])

            model = regularization_GlobalPointer(encoder,args.ent_type_size, 64)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion,batch_label_token):
        batch_size = input_ids.shape[0]

        num_models = len(self.models)
        outputs = []
        for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                token_type_ids=token_type_ids.to(self.device[i]),
                batch_labels=batch_labels.to(self.device[i]),
                criterion=criterion
            )

            output = tuple([o.to(0) for o in output])
            outputs.append(output)  # outputs里存放了 M个模型的输出

        loss_with_label = sum([output[0] for output in outputs]) / num_models
        model_output = outputs[0][1]  # 因为多个model的预测用kl散度进行靠近 所以返回第一个model的预测即可

        regularization_logits = [output[1] for output in outputs]  # model i

        ########################## #取mean_logits 作为软logits
        logits_stack_out = torch.stack(regularization_logits, dim=0)
        mean_logits_out = logits_stack_out.mean(0)
        mean_outputs_logits = (mean_logits_out,)
        ##########################

        from Code.run_ner.MRC_EG_G_Set_Config import bert_prefix_GlobalPointer_regularization_config
        args_ = bert_prefix_GlobalPointer_regularization_config()
        num_type = args_.ent_type_size

        import torch.nn.functional as F
        if args_.fit_type=="reg":
                # avg_logits.transpose(0, 1)
                # regularization_avg_logits = avg_logits.reshape(args_.ent_type_size,-1)  # [entity_type b s s]

                # logits_tran = [prob.transpose(0,1) for prob in probs1]
                # logits_ = [prob.reshape(args_.ent_type_size,-1) for prob in logits_tran]

                # token_input_label = (batch_label_token.view(-1) != -1).to(regularization_avg_logits)
                # reg_loss = sum([distance_(regularization_avg_logits, prob, args_) * token_input_label for prob in logits_]) / num_models

            avg_logits = mean_logits_out.reshape(-1)  #
            avg_logits = F.softmax(avg_logits,dim=0)  #
            logits_ = [prob.reshape(-1) for prob in regularization_logits]
            logits_ = [F.softmax(prob,dim=0) for prob in logits_]

            reg_loss = sum([distance_(avg_logits, prob, args_) for prob in logits_]) / num_models

            # reg_loss_sum = reg_loss.sum()
            # loss_out = loss_with_label + args_.alpha_t * reg_loss  # 联合训练的总损失
            loss_out = loss_with_label + reg_loss # 联合训练的总损失

        # model_output = (loss_out,) + mean_outputs_logits  #mean_logits 软标签
        models_output = (loss_out,) + (model_output,) #loss+kl + 第一个模块的标签

        return models_output

class create_bert_EfficientGlobalPointer_regularization(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))]  # 设备数

        for i in range(len(args.model_name_or_path)):
            encoder = Encoder_Bert(args.model_name_or_path[i])
            model = regularization_EfficientGlobalPointer(encoder,args.ent_type_size, 64)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion,batch_label_token):
        batch_size = input_ids.shape[0]

        num_models = len(self.models)
        outputs = []
        for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                token_type_ids=token_type_ids.to(self.device[i]),
                batch_labels=batch_labels.to(self.device[i]),
                criterion=criterion
            )

            output = tuple([o.to(0) for o in output])
            outputs.append(output)  # outputs里存放了 M个模型的输出

        loss_with_label = sum([output[0] for output in outputs]) / num_models
        model_output = outputs[0][1]  # 因为多个model的预测用kl散度进行靠近 所以返回第一个model的预测即可

        regularization_logits = [output[1] for output in outputs]  # model i

        ########################## #取mean_logits 作为软logits
        logits_stack_out = torch.stack(regularization_logits, dim=0)
        mean_logits_out = logits_stack_out.mean(0)
        mean_outputs_logits = (mean_logits_out,)
        ##########################

        from Code.run_ner.MRC_EG_G_Set_Config import bert_efficient_GlobalPointer_regularization_config
        args_ = bert_efficient_GlobalPointer_regularization_config()
        num_type = args_.ent_type_size

        import torch.nn.functional as F
        if args_.fit_type=="reg":
                # avg_logits.transpose(0, 1)
                # regularization_avg_logits = avg_logits.reshape(args_.ent_type_size,-1)  # [entity_type b s s]

                # logits_tran = [prob.transpose(0,1) for prob in probs1]
                # logits_ = [prob.reshape(args_.ent_type_size,-1) for prob in logits_tran]

                # token_input_label = (batch_label_token.view(-1) != -1).to(regularization_avg_logits)
                # reg_loss = sum([distance_(regularization_avg_logits, prob, args_) * token_input_label for prob in logits_]) / num_models

            avg_logits = mean_logits_out.reshape(-1)  #
            avg_logits = F.softmax(avg_logits,dim=0)  #
            logits_ = [prob.reshape(-1) for prob in regularization_logits]
            logits_ = [F.softmax(prob,dim=0) for prob in logits_]

            reg_loss = sum([distance_(avg_logits, prob, args_) for prob in logits_]) / num_models

            # reg_loss_sum = reg_loss.sum()
            # loss_out = loss_with_label + args_.alpha_t * reg_loss  # 联合训练的总损失
            loss_out = loss_with_label + reg_loss # 联合训练的总损失

        # model_output = (loss_out,) + mean_outputs_logits  #mean_logits 软标签
        models_output = (loss_out,) + (model_output,) #loss+kl + 第一个模块的标签

        return models_output

class create_bert_prefix_EfficientGlobalPointer_regularization(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(len(args.model_name_or_path))]  # 设备数

        for i in range(len(args.model_name_or_path)):
            encoder=Encoder_Bert_prefix(args.model_name_or_path[i])

            model = regularization_EfficientGlobalPointer(encoder,args.ent_type_size, 64)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, token_type_ids,batch_labels,criterion,batch_label_token):
        batch_size = input_ids.shape[0]

        num_models = len(self.models)
        outputs = []
        for i in range(num_models):  # 将input_id att_mask label 输入模型   得到模型[i]的输出
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                token_type_ids=token_type_ids.to(self.device[i]),
                batch_labels=batch_labels.to(self.device[i]),
                criterion=criterion
            )

            output = tuple([o.to(0) for o in output])
            outputs.append(output)  # outputs里存放了 M个模型的输出

        loss_with_label = sum([output[0] for output in outputs]) / num_models
        model_output = outputs[0][1]  # 因为多个model的预测用kl散度进行靠近 所以返回第一个model的预测即可

        regularization_logits = [output[1] for output in outputs]  # model i

        ########################## #取mean_logits 作为软logits
        logits_stack_out = torch.stack(regularization_logits, dim=0)
        mean_logits_out = logits_stack_out.mean(0)
        mean_outputs_logits = (mean_logits_out,)
        ##########################

        from Code.run_ner.MRC_EG_G_Set_Config import bert_prefix_efficient_GlobalPointer_regularization_config
        args_ = bert_prefix_efficient_GlobalPointer_regularization_config()
        num_type = args_.ent_type_size

        import torch.nn.functional as F
        if args_.fit_type=="reg":
                # avg_logits.transpose(0, 1)
                # regularization_avg_logits = avg_logits.reshape(args_.ent_type_size,-1)  # [entity_type b s s]

                # logits_tran = [prob.transpose(0,1) for prob in probs1]
                # logits_ = [prob.reshape(args_.ent_type_size,-1) for prob in logits_tran]

                # token_input_label = (batch_label_token.view(-1) != -1).to(regularization_avg_logits)
                # reg_loss = sum([distance_(regularization_avg_logits, prob, args_) * token_input_label for prob in logits_]) / num_models

            avg_logits = mean_logits_out.reshape(-1)  #
            avg_logits = F.softmax(avg_logits,dim=0)  #
            logits_ = [prob.reshape(-1) for prob in regularization_logits]
            logits_ = [F.softmax(prob,dim=0) for prob in logits_]

            reg_loss = sum([distance_(avg_logits, prob, args_) for prob in logits_]) / num_models

            # reg_loss_sum = reg_loss.sum()
            # loss_out = loss_with_label + args_.alpha_t * reg_loss  # 联合训练的总损失
            loss_out = loss_with_label + reg_loss # 联合训练的总损失

        # model_output = (loss_out,) + mean_outputs_logits  #mean_logits 软标签
        models_output = (loss_out,) + (model_output,) #loss+kl + 第一个模块的标签

        return models_output