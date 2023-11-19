import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    # [batch max_len]
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # 将input_id  att_mask  label 这三个列表的数据--------> 可以输入模型的 tensor
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return output


def collate_fn_span(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    # [batch max_len]  进行pading
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    token_type_ids = [[0] * len(f["input_ids"]) + [0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]
    token_label = [f["label_token"] + [-1] * (max_len - len(f["label_token"])) for f in batch]
    start_ids = [f["start_ids"] + [0] * (max_len - len(f["start_ids"])) for f in batch]
    end_ids = [f["end_ids"] + [0] * (max_len - len(f["end_ids"])) for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # 将input_id  att_mask  label 这三个列表的数据--------> 可以输入模型的 tensor
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    token_label = torch.tensor(token_label, dtype=torch.long)
    start_ids = torch.tensor(start_ids, dtype=torch.long)
    end_ids = torch.tensor(end_ids, dtype=torch.long)

    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids':token_type_ids,
        'labels': labels,
        'token_label': token_label,
        'start_ids': start_ids,
        'end_ids': end_ids,
    }
    return output


def collate_fn_span_have_ans(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    # max_len = max_seqlen

    # [batch max_len]  进行pading
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]
    token_label = [f["label_token"] + [-1] * (max_len - len(f["label_token"])) for f in batch]
    start_ids = [f["start_ids"] + [0] * (max_len - len(f["start_ids"])) for f in batch]
    end_ids = [f["end_ids"] + [0] * (max_len - len(f["end_ids"])) for f in batch]
    have_ans = [f["have_enti"] for f in batch]
    ans_nums = [f["enti_num"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # 将input_id  att_mask  label 这三个列表的数据--------> 可以输入模型的 tensor
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    token_label = torch.tensor(token_label, dtype=torch.long)
    start_ids = torch.tensor(start_ids, dtype=torch.long)
    end_ids = torch.tensor(end_ids, dtype=torch.long)
    have_ans = torch.tensor(have_ans, dtype=torch.long)
    ans_nums = torch.tensor(ans_nums, dtype=torch.long)

    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        # 'labels': labels,
        'token_label': token_label,
        'start_ids': start_ids,
        'end_ids': end_ids,
        'have_ans': have_ans,
        'ans_nums': ans_nums,
    }
    return output


def collate_fn_soft_prompt(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    # [batch max_len]
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    token_labels = [f["labels"] for f in batch]
    length_token = len(token_labels)

    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # 将input_id  att_mask  label 这三个列表的数据--------> 可以输入模型的 tensor
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'length_token': length_token
    }
    return output
