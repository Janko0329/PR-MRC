
####################################################
NCBI_BC5CDR_dise_crf_LABEL_TO_ID = {'O': 0, 'B-Disease': 1, 'I-Disease': 2 ,"[START]":3, "[END]":4}
NCBI_BC5CDR_dise_LABEL_TO_ID = {'O': 0, 'B-Disease': 1, 'I-Disease': 2}
NCBI_BC5CDR_dise_SPAN_LABEL_TO_ID = {'O': 0, 'Disease': 1}
ENT2ID_NCBI_BC5CDR_dise = {"0": 0, "1": 1}

BC2GM_crf_LABEL_TO_ID = {'O': 0, 'B-GENE': 1, 'I-GENE': 2 ,"[START]":3, "[END]":4}
BC2GM_LABEL_TO_ID = {'O': 0, 'B-GENE': 1, 'I-GENE': 2}
BC2GM_SPAN_LABEL_TO_ID = {'O': 0, 'GENE': 1}
ENT2ID_BC2GM = {"0": 0, "1": 1}

BC5CDR_chem_crf_LABEL_TO_ID = {'O': 0, 'B-Chemical': 1, 'I-Chemical': 2 ,"[START]":3, "[END]":4}
BC5CDR_chem_LABEL_TO_ID = {'O': 0, 'B-Chemical': 1, 'I-Chemical': 2}
BC5CDR_chem_SPAN_LABEL_TO_ID = {'O': 0, 'Chemical': 1}
ENT2ID_BC5CDR_chem = {"0": 0, "1": 1}

JNLPBA_crf_LABEL_TO_ID = {'O':0,'B-protein':1,'I-protein':2, 'B-DNA':3,'I-DNA':4 , 'B-cell_type':5, 'I-cell_type':6, 'B-cell_line':7, 'I-cell_line':8, 'B-RNA':9, 'I-RNA':10,"[START]":11, "[END]":12}
JNLPBA_LABEL_TO_ID = {'O':0,'B-protein':1,'I-protein':2, 'B-DNA':3,'I-DNA':4 , 'B-cell_type':5, 'I-cell_type':6, 'B-cell_line':7, 'I-cell_line':8, 'B-RNA':9, 'I-RNA':10}
JNLPBA_SPAN_LABEL_TO_ID = {'O':0,'protein':1, 'DNA':2, 'cell_type':3, 'cell_line':4, 'RNA':5}
ENT2ID_JNLPBA = {"0": 0, "1": 1, "2":2, "3":3, "4":4, "5":5}

#修改数据集
data_set_="NCBI-disease"  # NCBI-disease  BC5CDR_disease  20/14
crf_LABEL_TO_ID= NCBI_BC5CDR_dise_crf_LABEL_TO_ID
LABEL_TO_ID = NCBI_BC5CDR_dise_LABEL_TO_ID
SPAN_LABEL_TO_ID = NCBI_BC5CDR_dise_SPAN_LABEL_TO_ID
ent2id = ENT2ID_NCBI_BC5CDR_dise

# data_set_="BC2GM"  #BC2GM
# crf_LABEL_TO_ID= BC2GM_crf_LABEL_TO_ID
# LABEL_TO_ID = BC2GM_LABEL_TO_ID
# SPAN_LABEL_TO_ID = BC2GM_SPAN_LABEL_TO_ID
# ent2id = ENT2ID_BC2GM

# data_set_="BC5CDR_chem"  # BC5CDR_chem  ,14
# crf_LABEL_TO_ID= BC5CDR_chem_crf_LABEL_TO_ID
# LABEL_TO_ID = BC5CDR_chem_LABEL_TO_ID
# SPAN_LABEL_TO_ID = BC5CDR_chem_SPAN_LABEL_TO_ID
# ent2id = ENT2ID_BC5CDR_chem

# data_set_="JNLPBA"  #JNLPBA
# crf_LABEL_TO_ID= JNLPBA_crf_LABEL_TO_ID
# LABEL_TO_ID = JNLPBA_LABEL_TO_ID
# SPAN_LABEL_TO_ID = JNLPBA_SPAN_LABEL_TO_ID
# ent2id = ENT2ID_JNLPBA