import truecase
import re
from Code.run_ner.Set_Config import LABEL_TO_ID,crf_LABEL_TO_ID,SPAN_LABEL_TO_ID,data_set_


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

def get_entity_bio(text_seq,seq,id2label):
    chunks = []
    entity_span=[]
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
    temp=chunks
    x1=[st[1] for st in temp]
    x2=[ed[2] for ed in temp]
    entity_label=[ed[0] for ed in temp]

    entity_word_sp_dict={}
    entity_type_span_dict={}
    data_dict={}
    text_seq_words = str(' '.join((str(i) for i in text_seq)))
    text_len=len(text_seq_words.split(" "))

    for i in range(len(x1)):
        entity_word=text_seq[x1[i]:x2[i]+1]

        entity_words=str(' '.join((str(i) for i in entity_word)))
        entity_type_word=entity_label[0] #标签都是B-x,I-x  相当于只有一种实体类型
        entity_span_list=[[x1[i],x2[i]]]

        entity_word_sp_dict[entity_words]=entity_span_list
        entity_type_span_dict[entity_type_word]=entity_word_sp_dict
        data_dict["text"] = text_seq_words #文本
        data_dict["label"] = entity_type_span_dict #标签
    return chunks,data_dict,text_len

def get_entities(text_seq,seq,id2label,markup='bio'):
    if markup =='bio':
        # return get_entity_bio(seq,id2label)
        return get_entity_bio(text_seq, seq, id2label)

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
                    subjects_,process_data ,len_text= get_entities(words ,labels, id2label=None, markup='bio')
                    examples.append(process_data)
                    words, labels = [], []
    return examples

def write_file(path,data):
    import os
    import json
    with open(path,'w',encoding='utf8') as f:
        for text_data in data:
            if(text_data!={}):
                # f.write(str(text_data))
                f.write(json.dumps(text_data))
                f.write('\n')
    f.close() #关闭文件流

if __name__ == '__main__':
    from transformers import AutoTokenizer

    train_file = './data/Data_set/NCBI-disease/test.tsv'
    process_data_path = './data/Data_set/NCBI-disease/test.json'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    # read_ner_data_span(train_file, tokenizer, max_seq_length=512)
    train_data = read_ner_data_span(train_file, tokenizer, max_seq_length=512)
    write_file(process_data_path,train_data)
    print(train_data[0:1])