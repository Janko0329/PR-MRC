import truecase
import re
from Code.run_ner.MRC_Pointer_Set_Config import LABEL_TO_ID

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
        # for word, label in words, labels:
        tokenized = tokenizer.tokenize(word)
        token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)
        # token_label = [LABEL_TO_ID[label]] + [0] * (len(tokenized) - Data_set)
        tokens += tokenized
        token_labels += token_label

    assert len(tokens) == len(token_labels)
    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    token_labels = [-1] + token_labels + [-1]
    return {
        "input_ids": input_ids,
        "labels": token_labels
    }

# Manual template
# {in this sentence :}+{text}+{recognition all disease , such as :}
#     ["in","this","sentence",":",] ___["recognition","all","disease",",","such","as",":",]
#   [0 0 0 0]+{text_label}+[0 0 0 0 0 0 0]+{label}
def read_manual_emplate_data(file_in, tokenizer, max_seq_length=512):  # 读数据
    words, labels = [], []
    word_list, label_list = [], []
    examples = []
    is_title = False

    with open(file_in, "r") as fh:
        # fh = f.readline()
        for line in fh:
            t0 = ["in", "this", "sentence", ":", ]
            t0_l=['O', 'O', 'O', 'O']
            t1 = ["recognition", "all", "disease", "."]
            t1_l = ['O', 'O', 'O', 'O']
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                is_title = True
                continue
            if len(line) > 0:
                line = line.split()
                # print(line[-Data_set])
                # break
                word = line[0]  # 将读行的 第一个字符给word
                label = line[-1]  # 将读行的 最后一个token 给label

                if line[-1] != 'O':
                    word_list.append(line[0])
                    label_list.append(line[-1])

                words.append(word)
                labels.append(label)

            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False

                    # words = t0+words+t1+word_list
                    # labels = t0_l+labels+t1_l+label_list
                    words = t0 + words + t1
                    labels = t0_l + labels + t1_l
                    # print(words)
                    # print(len(words))
                    # print(labels)
                    # print(len(labels))
                    # print(word_list)
                    # print(label_list)
                    assert len(words) == len(labels)
                    assert len(word_list) == len(label_list)
                    examples.append(process_instance(words, labels, tokenizer, max_seq_length))
                    words, labels = [], []
                    word_list, label_list = [], []

    return examples

def read_test_data(file_in, tokenizer, max_seq_length=512):  #读数据
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
                word = line[0]  #将读行的 第一个字符给word
                label = line[-1] #将读行的 最后一个token 给label
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
# if __name__ == '__main__':
#     from transformers import AutoTokenizer
#     train_file = './prefix_len-disease-IOB/train.txt'
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
#     x = read_ner_data(train_file, tokenizer, max_seq_length=512)
#     train_features = read_ner_data(train_file, tokenizer, max_seq_length=512)
#     print(x[0:10])