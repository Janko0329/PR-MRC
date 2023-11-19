import truecase
import re

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

def read_and_get_label(file_in, tokenizer, max_seq_length=512):
    words, labels = [], []
    set_label = []
    examples = []
    te =[]
    la= []
    is_title = False
    with open(file_in, "r",encoding='utf8') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                is_title = True
                continue
            if len(line) > 0:
                line = line.split()
                word = line[0]
                label = line[-1]
                words.append(word)
                labels.append(label)
                set_label.append(label)

            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False
                    assert len(words) == len(labels)
                    te.append(words)
                    la.append(labels)

                    words, labels = [], []
    print(list(set(set_label)))


    # return examples
    print("xxxxxxxxxxxxxxxxxxxxx")
    te_train = te[0:int(len(te)*0.9)]
    te_dev = te[int(len(te) * 0.9):len(te)]
    la_train = la[0:int(len(la)*0.9)]
    la_dev = la[int(len(la) * 0.9):len(la)]
    with open('./NCBI-disease/train.tsv','w',encoding='utf8') as f1:
        for t,l in zip(te_train,la_train):
            for tt,ll in zip(t,l):
                f1.write(tt)
                f1.write("\t")
                f1.write(ll)
                f1.write('\n')
            f1.write('\n')  #写完一句 换行

    with open('./NCBI-disease/dev.tsv','w',encoding='utf8') as f2:
        for t,l in zip(te_dev,la_dev):
            for tt,ll in zip(t,l):
                f2.write(tt)
                f2.write("\t")
                f2.write(ll)
                f2.write('\n')
            f2.write('\n')  #写完一句 换行

if __name__ == '__main__':
    from transformers import AutoTokenizer
    train_file = 'NCBI-disease/train_initial.tsv'
    tokenizer = AutoTokenizer.from_pretrained('D:\\PY\\pycharm_code\\embedding\\bert-base-cased')
    read_and_get_label(train_file, tokenizer, max_seq_length=512)




