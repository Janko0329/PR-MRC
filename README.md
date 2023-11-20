# PR-MRC / EGP-MRC
Source codes and data for Paper: Span-based Model for Biomedical Named Entity Recognition

## Data
The dataset required for the experiment has been placed ```/read_data/data/Data_set/```.

You can use your own dataset for substitution.

## Run
The code program is in the ```/run_ner/```.

The code for EGP-MRC and PR-MRC can be run separately, and the parameters have been set.

You can change the parameters within the file ```data_set_config.py```, ```MRC_EG_G_Set_config.py``` and `MRC_Pointer_Set_Config.py`

For Example, 

1. run PR-MRC:
```shell
python /run_ner/MRC_GlobalPointer/run_bert_prefix_globalpointer_regularization.py
```

2. run PR-MRC w/o prefix:
```shell
python /run_ner/MRC_GlobalPointer/run_bert_globalpointer_regularization.py
```

3. run PR-MRC w/o regularization:
```shell
python /run_ner/MRC_GlobalPointer/run_bert_prefix_globalpointer.py
```

4. run EGP-MRC:
```shell
python /run_ner/MRC_EfficientGlobalPointer/run_bert_prefix_efficient_globalpointer_regularization.py
```
5. run EGP-MRC w/o prefix:
```shell
python /run_ner/MRC_EfficientGlobalPointer/run_bert_globalpointer_regularization.py
```

6. run EGP-MRC w/o regularization:
```shell
python /run_ner/MRC_EfficientGlobalPointer/run_bert_prefix_efficient_globalpointer.py
```

## Model
You can download our trained model at the following link.

|         | NCBI-Disease               | BC2GM | BC5CDR-Chem |
|---------|----------------------------|-------|-------------|
| PR-MRC  | [checkpoints](https://drive.google.com/file/d/1mGbYCg8S45PQGGtbh8BnPxnexZkyEP27/view?usp=sharing) | [checkpoints]( )|[checkpoints]( )|
| EGP-MRC |[checkpoints]( ) | [checkpoints]( )|[checkpoints]( )|

