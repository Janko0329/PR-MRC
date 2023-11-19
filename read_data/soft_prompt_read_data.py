import truecase
import re
from Code.run_ner.Set_Config import crf_LABEL_TO_ID,SPAN_LABEL_TO_ID,LABEL_TO_ID

# 用正则 清洗数据
def true_case(tokens):
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        if len(parts) != len(word_lst):
            return tokens
        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens


# 获得input_ids
def process_instance(words, labels, tokenizer, max_seq_length=512):
    tokens, token_labels = [], []
    for word, label in zip(words, labels):
        tokenized = tokenizer.tokenize(word)
        token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)

        tokens += tokenized
        token_labels += token_label

    assert len(tokens) == len(token_labels)
    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) #cls sep
    ##############################
    import torch
    input_ids = torch.tensor([input_ids])
    input_ids = torch.cat([input_ids,torch.full((1, 20), 28996)], 1)
    # input_ids = torch.cat([torch.full((Data_set, 20), 28996), input_ids], Data_set)
    input_ids = input_ids.numpy().tolist()
    ##################################

    token_labels = [-1] + token_labels + [-1]

    return {
        "input_ids": input_ids,
        "labels": token_labels
    }


def read_data(file_in, tokenizer, max_seq_length=512):  # 读数据
    words, labels = [], []
    examples = []
    is_title = False
    with open(file_in, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                is_title = True
                continue
            if len(line) > 0:
                line = line.split()
                word = line[0]  # 将读行的 第一个字符给word
                label = line[-1]  # 将读行的 最后一个token 给label
                words.append(word)
                labels.append(label)

            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False

                    assert len(words) == len(labels)
                    examples.append(process_instance(words, labels, tokenizer, max_seq_length))
                    words, labels = [], []

    return examples
