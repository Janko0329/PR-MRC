import truecase
import re

from Code.run_ner.data_set_config import LABEL_TO_ID,crf_LABEL_TO_ID,SPAN_LABEL_TO_ID,data_set_,ent2id

#用正则 清洗数据
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


def process_instance(words, labels, tokenizer, max_seq_length=512):
    tokens, token_labels = [], []
    for word, label in zip(words, labels):
        tokenized = tokenizer.tokenize(word)
        token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)  #-Data_set 是因为LABEL_TO_ID[label] 匹配了一个

        tokens += tokenized
        token_labels += token_label
    assert len(tokens) == len(token_labels)


    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]  # -2 是因为要加cls 和 sep
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) #[cls token sep]
    
    token_labels = [-1] + token_labels + [-1]  #[-Data_set id -Data_set]
    return {
        "input_ids": input_ids,
        "labels": token_labels
    }

def read_ner_data(file_in, tokenizer, max_seq_length=512):  #读数据
    words, labels = [], []
    examples = []
    is_title = False
    with open(file_in, "r" ,encoding='utf8') as fh:
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

def get_entity_bio(seq,id2label):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('B-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bio'):
    if markup =='bio':
        return get_entity_bio(seq,id2label)

# bert_span 读数据
def process_instance_span(words, labels, tokenizer,subjects_,max_seq_length=512):
    tokens, token_labels,token_labels_ = [], [],[]

    words_=str(' '.join((str(i) for i in words)))
    for word, label in zip(words, labels):
        # for word, label in words, labels:
        tokenized = tokenizer.tokenize(word)

        TEMP = label.replace("B-", "")
        TEMP1=(TEMP.replace("I-", ""))
        token_l = [SPAN_LABEL_TO_ID[TEMP1]] + [-1] * (len(tokenized) - 1)  #-Data_set 是因为LABEL_TO_ID[label] 匹配了一个

        if data_set_ == "NCBI-disease" or data_set_ == "BC5CDR_disease":
            if label == 'O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-Disease':
                token_label = [label] + ['I-Disease'] * (len(tokenized) - 1)
            elif label == 'I-Disease':
                token_label = [label] + ['I-Disease'] * (len(tokenized) - 1)

        elif data_set_ == "BC5CDR_chem":
            if label == 'O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-Chemical':
                token_label = [label] + ['I-Chemical'] * (len(tokenized) - 1)
            elif label == 'I-Chemical':
                token_label = [label] + ['I-Chemical'] * (len(tokenized) - 1)

        elif data_set_ == "BC2GM":
            if label == 'O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-GENE':
                token_label = [label] + ['I-GENE'] * (len(tokenized) - 1)
            elif label == 'I-GENE':
                token_label = [label] + ['I-GENE'] * (len(tokenized) - 1)

        elif data_set_ == "JNLPBA":
            if label == 'O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label]
            elif label == 'B-protein':
                token_label = [label] + ['I-protein'] * (len(tokenized) - 1)
            elif label == 'I-protein':
                token_label = [label] + ['I-protein'] * (len(tokenized) - 1)
            elif label == 'B-DNA':
                token_label = [label] + ['I-DNA'] * (len(tokenized) - 1)
            elif label == 'I-DNA':
                token_label = [label] + ['I-DNA'] * (len(tokenized) - 1)
            elif label == 'B-cell_type':
                token_label = [label] + ['I-cell_type'] * (len(tokenized) - 1)
            elif label == 'I-cell_type':
                token_label = [label] + ['I-cell_type'] * (len(tokenized) - 1)
            elif label == 'B-cell_line':
                token_label = [label] + ['I-cell_line'] * (len(tokenized) - 1)
            elif label == 'I-cell_line':
                token_label = [label] + ['I-cell_line'] * (len(tokenized) - 1)
            elif label == 'B-RNA':
                token_label = [label] + ['I-RNA'] * (len(tokenized) - 1)
            elif label == 'I-RNA':
                token_label = [label] + ['I-RNA'] * (len(tokenized) - 1)

        tokens += tokenized
        token_labels += token_label
        token_labels_ +=token_l

    assert len(tokens) == len(token_labels)
    # assert len(tokens) == len(token_labels_)

    start_ids = [0] * len(tokens)
    end_ids = [0] * len(tokens)

    subjects = get_entities(token_labels, id2label=None, markup='bio')
    subjects_id = []
    for subject in subjects:
        label = subject[0]
        start = subject[1]
        end = subject[2]
        start_ids[start] = SPAN_LABEL_TO_ID[label]  # 将标签转id值 对应位置的 start_id 数组标记为起点
        end_ids[end] = SPAN_LABEL_TO_ID[label]
        subjects_id.append((SPAN_LABEL_TO_ID[label], start, end))  # 将标签的ID start 和 end 加入


    # subjects_id_ = []
    # for subject in subjects_:
    #     label = subject[0]
    #     start = subject[Data_set]
    #     end = subject[2]
    #     start_ids[start] = SPAN_LABEL_TO_ID[label]  # 将标签转id值 对应位置的 start_id 数组标记为起点
    #     end_ids[end] = SPAN_LABEL_TO_ID[label]
    #     subjects_id_.append((SPAN_LABEL_TO_ID[label], start, end))  # 将标签的ID start 和 end 加入

    tokens, token_labels,token_labels_ = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2] , token_labels_[:max_seq_length - 2] # -2 是因为要加cls 和 sep
    # tokens, token_labels_ = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]  # -2 是因为要加cls 和 sep
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)  # [cls token sep]

    inputs = tokenizer(
        words_,
        truncation=True,
    )

    token_labels = ['O'] + token_labels + ['O']  # [0 id 0]
    token_labels_ = [0] + token_labels_ + [0]  # [0 id 0]

    start_ids = [0] + start_ids + [0]
    end_ids = [0] + end_ids + [0]
    input_mask = [1] * len(input_ids)
    token_type_ids = [0] *len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": token_labels,
        "label_token":token_labels_,
        "start_ids":start_ids,
        "end_ids" : end_ids,
        "subjects":subjects_id,
        # "subjects_": subjects_id_,
        "input_mask":input_mask,
        "token_type_ids":token_type_ids,
    }

#softmax读数据
def read_ner_data_span(file_in, tokenizer, max_seq_length=512):  # 读数据
    words, labels = [], []
    examples = []
    is_title = False
    with open(file_in, "r", encoding='utf8') as fh:
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

                # TEMP = label.replace("B-", "")
                # labels.append(TEMP.replace("I-", ""))
            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False

                    assert len(words) == len(labels)

                    subjects_ = get_entities(labels, id2label=None, markup='bio')
                    examples.append(process_instance_span(words, labels, tokenizer,subjects_, max_seq_length))
                    words, labels = [], []
    return examples

def process_instance_soft_prompt(words, labels, tokenizer, max_seq_length=512):
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
    input_ids = input_ids.numpy().tolist()[0]
    ##################################

    token_labels = [-1] + token_labels + [-1]

    return {
        "input_ids": input_ids,
        "labels": token_labels
    }

def read_data_soft_prompt(file_in, tokenizer, max_seq_length=512):  # 读数据
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
                    examples.append(process_instance_soft_prompt(words, labels, tokenizer, max_seq_length))
                    words, labels = [], []

    return examples

def process_instance_span_have_ans(words, labels, tokenizer,subjects_,hava_entity_,entity_count,max_seq_length=512):
    tokens, token_labels,token_labels_ ,sentence_label1,sentence_label2= [], [],[],[],[]
    for word, label in zip(words, labels):
        # for word, label in words, labels:
        tokenized = tokenizer.tokenize(word)

        TEMP = label.replace("B-", "")
        TEMP1 = (TEMP.replace("I-", ""))
        token_l = [SPAN_LABEL_TO_ID[TEMP1]] + [-1] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个

        if data_set_ =="NCBI-disease" or data_set_ =="BC5CDR_disease":
            if label=='O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-Disease':
                token_label = [label] + ['I-Disease'] * (len(tokenized) - 1)
            elif label == 'I-Disease':
                token_label = [label] + ['I-Disease'] * (len(tokenized) - 1)

        elif data_set_ =="BC4CHEMD" or data_set_ =="BC5CDR_chem":
            if label=='O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-Chemical':
                token_label = [label] + ['I-Chemical'] * (len(tokenized) - 1)
            elif label == 'I-Chemical':
                token_label = [label] + ['I-Chemical'] * (len(tokenized) - 1)

        elif data_set_ =="BC2GM":
            if label=='O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label] 匹配了一个
            elif label == 'B-GENE':
                token_label = [label] + ['I-GENE'] * (len(tokenized) - 1)
            elif label == 'I-GENE':
                token_label = [label] + ['I-GENE'] * (len(tokenized) - 1)

        elif data_set_ == "JNLPBA":
            if label=='O':
                token_label = [label] + [label] * (len(tokenized) - 1)  # -Data_set 是因为LABEL_TO_ID[label]
            elif label == 'B-protein':
                token_label = [label] + ['I-protein'] * (len(tokenized) - 1)
            elif label == 'I-protein':
                token_label = [label] + ['I-protein'] * (len(tokenized) - 1)
            elif label == 'B-DNA':
                token_label = [label] + ['I-DNA'] * (len(tokenized) - 1)
            elif label == 'I-DNA':
                token_label = [label] + ['I-DNA'] * (len(tokenized) - 1)
            elif label == 'B-cell_type':
                token_label = [label] + ['I-cell_type'] * (len(tokenized) - 1)
            elif label == 'I-cell_type':
                token_label = [label] + ['I-cell_type'] * (len(tokenized) - 1)
            elif label == 'B-cell_line':
                token_label = [label] + ['I-cell_line'] * (len(tokenized) - 1)
            elif label == 'I-cell_line':
                token_label = [label] + ['I-cell_line'] * (len(tokenized) - 1)
            elif label == 'B-RNA':
                token_label = [label] + ['I-RNA'] * (len(tokenized) - 1)
            elif label == 'I-RNA':
                token_label = [label] + ['I-RNA'] * (len(tokenized) - 1)

        tokens += tokenized
        token_labels += token_label
        token_labels_ +=token_l

    assert len(tokens) == len(token_labels)
    # assert len(tokens) == len(token_labels_)
    sentence_label1.append(hava_entity_) #是否有实体的二分类
    sentence_label2.append(entity_count) #有几个实体的多分类

    start_ids = [0] * len(tokens)
    end_ids = [0] * len(tokens)

    subjects = get_entities(token_labels, id2label=None, markup='bio')
    subjects_id = []
    for subject in subjects:
        label = subject[0]
        start = subject[1]
        end = subject[2]
        start_ids[start] = SPAN_LABEL_TO_ID[label]  # 将标签转id值 对应位置的 start_id 数组标记为起点
        end_ids[end] = SPAN_LABEL_TO_ID[label]
        subjects_id.append((SPAN_LABEL_TO_ID[label], start, end))  # 将标签的ID start 和 end 加入


    # subjects_id_ = []
    # for subject in subjects_:
    #     label = subject[0]
    #     start = subject[Data_set]
    #     end = subject[2]
    #     start_ids[start] = SPAN_LABEL_TO_ID[label]  # 将标签转id值 对应位置的 start_id 数组标记为起点
    #     end_ids[end] = SPAN_LABEL_TO_ID[label]
    #     subjects_id_.append((SPAN_LABEL_TO_ID[label], start, end))  # 将标签的ID start 和 end 加入

    tokens, token_labels,token_labels_ = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2] , token_labels_[:max_seq_length - 2]  # -2 是因为要加cls 和 sep
    # tokens, token_labels_ = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]  # -2 是因为要加cls 和 sep
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)  # [cls token sep]

    token_labels = ['O'] + token_labels + ['O']  # [0 id 0]
    token_labels_ = [0] + token_labels_ + [0]  # [0 id 0]

    start_ids = [0] + start_ids + [0]
    end_ids = [0] + end_ids + [0]
    input_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": token_labels,
        "label_token": token_labels_,
        "start_ids":start_ids,
        "end_ids" : end_ids,
        "subjects":subjects_id,
        # "subjects_": subjects_id_,
        "input_mask":input_mask,
        "have_enti":sentence_label1,
        "enti_num":sentence_label2
    }

entity_num_max=0
def read_ner_data_span_have_ans(file_in, tokenizer, max_seq_length=512):  # 读数据
    words, labels = [], []
    examples = []
    entity_count = 0
    is_title = False
    with open(file_in, "r", encoding='utf8') as fh:
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

                    for la in labels:
                        if la == "B-Disease":
                            entity_count += 1
                            # entity_num_max = max(entity_count, entity_num_max)
                        if entity_count > 0:
                            hava_entity_ = 1
                        elif entity_count==0:
                            hava_entity_ = 0

                    assert len(words) == len(labels)

                    subjects_ = get_entities(labels, id2label=None, markup='bio')
                    examples.append(process_instance_span_have_ans(words, labels, tokenizer,subjects_,hava_entity_,entity_count, max_seq_length))
                    words, labels = [], []
                    hava_entity_ = 0
                    entity_count = 0
    return examples


# if __name__ == '__main__':
#     from transformers import AutoTokenizer
#     # train_file = '../data/train.txt'
#     train_file = '../data/NER-DataSet/prefix_len-disease-IOB/train.txt'
#     tokenizer = AutoTokenizer.from_pretrained('../model_embedding/bert-base-cased')
#     read_ner_data(train_file, tokenizer, max_seq_length=512)
#     train_features = read_ner_data(train_file, tokenizer, max_seq_length=512)
#     print(train_features[0:10])