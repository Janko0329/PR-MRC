
import torch

class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    #entity_list eg: [(0, 3, 'book'), (17, 19, 'name'), (21, 22, 'name')]
    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans
        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []

        #intpus 将text进行分词 并进行映射 返回input_id att_mask token_type_id offset_map
        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)

        token2char_span_mapping = inputs["offset_mapping"]

        #cls 分词 sep
        list_texts = text.split(" ")
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)
        # text2tokens = self.tokenizer.tokenize(list_texts, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:  #eg  ent_span:(21, 22, 'name')

            ent = list_texts[ent_span[0]:ent_span[1] + 1]  #start和end跨度内的实体词 eg:'石齐'
            # ent2token=[]
            # for x in ent:
            #     temp_token = self.tokenizer.tokenize(x, add_special_tokens=False) #将实体词进行分词  eg：['石', '齐']
            #     ent2token.append(temp_token)

            ent_words = str(' '.join((str(i) for i in ent)))  #将实体词 拼接成字符串
            ent2token = self.tokenizer.tokenize(ent_words, add_special_tokens=False)  # 将实体词进行分词  eg：['石', '齐']

            # 寻找ent的token_span  按照收尾token 进行匹配
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]  #[22] 由于加了cls 所以从21->22
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]   #[23]

            # token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            # token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1], token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

            # if len(token_start_index) == 0 or len(token_end_index) == 0:
            #     # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
            #     continue

            # # eg  按照wordpice分词后的 实体_span:(22, 23, 'name')
            # token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            # ent2token_spans.append(token_span)

            if len(token_start_indexs) == 0 or len(token_end_indexs) == 0: #0为cls的位置 配对到0 肯定是匹配失败
                # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue

            # eg  按照wordpice分词后的 实体_span:(22, 23, 'name')
            token_span = (token_start_indexs[0], token_end_indexs[0], ent_span[2])
            ent2token_spans.append(token_span)

        return ent2token_spans
