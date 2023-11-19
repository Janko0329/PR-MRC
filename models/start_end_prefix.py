import torch
import torch.nn as nn


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)  # [b s 2]
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, ):
        # x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-Data_set)) #[b s 768] [b s 2]-->[b s 770]
        x = self.dense_0(torch.cat([start_positions, hidden_states], dim=-1))  # [b s 768] [b s 2]-->[b s 770]
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


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


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]

    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    S_final = []
    if len(S) < 2:
        S_final = S
    else:
        for i in range(len(S) - 1):
            if S[i][2] < S[i + 1][1]:
                S_final.append(S[i])
        S_final.append(S[len(S) - 1])
    return S_final

# def process_span(sp,length_):
#     global count
#     if sp==[]:
#         return ['O'] * length_
#     else:
#         processed_span = ['O'] * length_
#         for pre in sp:
#             for i in range(pre[1], pre[2] + 1):
#                 processed_span[i] = SPAN_ID_TO_LABEL[pre[0]]
#         return processed_span
