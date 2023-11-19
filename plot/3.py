import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import truecase
import re
import statistics

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


def read_data(file_in):
    data_len = []
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
                word = line[0]
                label = line[-1]
                words.append(word)
                labels.append(label)
            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False
                    assert len(words) == len(labels)
                    data_len.append(len(words))
                    words, labels = [], []
    return data_len

#作图1 NCBI
plt.subplot(2,3,1)
p_len = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
# pl.xlim(2, 26)  # set axis limits
plt.xticks(p_len)
pl.ylim(0.89, 0.92)

p_len = [3, 5, 7, 9, 11, 13, 15 ,17,19,21,23,25]
f1 = [0.90134994807892, 0.902892561983471, 0.9042663891779397, 0.9031589849818746, 0.9073120494335737, 0.902145473574045, 0.9041794087665647,0.904688304997424,0.9030927835051547,0.9006276150627616,0.9,0.8947641264904097]
prec = [0.8985507246376812, 0.9004166666666667, 0.9033264033264033, 0.898043254376931, 0.8971486761710794, 0.9064143007360673, 0.9007763007361352,0.8950050968399592,0.8938775510204081,0.9044117647058824,0.9,0.890608875128999]
reca = [0.9041666666666667, 0.9053821108903902, 0.9052083333333333, 0.9083333333333333, 0.9177083333333333, 0.8979166666666667, 0.9076083280076720,0.9145833333333333,0.9125,0.896875,0.9,0.8989583333333333]

pl.plot(p_len, f1, 'plum', label='f1', linestyle='-', marker='.')  # use pylab to plot x and y : Give your plots names
pl.plot(p_len, prec, 'cornflowerblue', label='precision', linestyle='-.', marker='.')
pl.plot(p_len, reca, 'lightskyblue', label='recall', linestyle='--', marker='.')

pl.xlabel('Prefixes length for NCBI-Disease')  # make axis labels
pl.ylabel('Performance')
pl.legend()

#作图2 bc2gm
plt.subplot(2,3,2)
p_len = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
# pl.xlim(2, 26)  # set axis limits
plt.xticks(p_len)
pl.ylim(0.845, 0.875)

p_len = [3, 5, 7, 9, 11, 13,15,17,19,21,23,25]
f1 = [0.8573025632835581, 0.8586253692025225, 0.8578454704109154, 0.8587112171837707, 0.8576653944020356,0.8578972635621701,0.859541130386122,0.8610332398878654,0.8568469688520672,0.8597342668470046,0.8575997441432798,0.860392967942089]
prec = [0.8660858341400451, 0.8671396323766527, 0.8608501830918643, 0.8642113690952762, 0.8627419612861942,0.8684594200550786,0.869340232858991,0.872564935064935,0.863519588953115,0.8653106982703396,0.8675186023940472,0.8658341338456612]
reca = [0.8486956521739131, 0.8502766798418973, 0.8548616600790514, 0.8532806324110672, 0.8526482213438735,0.8475889328063241,0.8499604743083004,0.849802371541502,0.8502766798418973,0.8542292490118577,0.847905138339921,0.8550197628458498]

pl.plot(p_len, f1, 'plum', label='f1', linestyle='-', marker='.')  # use pylab to plot x and y : Give your plots names
pl.plot(p_len, prec, 'cornflowerblue', label='precision', linestyle='-.', marker='.')
pl.plot(p_len, reca, 'lightskyblue', label='recall', linestyle='--', marker='.')

# pl.title('Effect of Prefix Sequence Length')  # give plot a title
pl.xlabel('Prefixes length for BC2GM')  # make axis labels
# pl.ylabel('Performance')
# pl.legend()

#作图3 bc5 chme
plt.subplot(2,3,3)
p_len = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
# pl.xlim(2, 26)  # set axis limits
plt.xticks(p_len)
pl.ylim(0.93, 0.95)

f1 = [0.9381289601192694,0.9383453036097379,0.9402249697871154,0.9400749063670413,0.9421084830153559,0.9421965317919077,0.9398510242085661,0.940093023255814,0.9389836795252225,0.9389497123770644,0.9387337057728119,0.9397658079625293]
prec = [0.9414625023377595,0.9426536731634183,0.9413626209977662,0.9480642115203022,0.9449253731343283,0.9460775135742371,0.9424836601307189,0.9418452935694315,0.9377662530098166,0.9382532913035416,0.9413632119514472,0.9482041587901702]
reca = [0.9348189415041783,0.9340761374187558,0.9390900649953575,0.9322191272051996,0.9393679834335462,0.938347260909935,0.9372330547818013,0.938347260909935,0.9402042711234911,0.9396471680594243,0.9361188486536676,0.9314763231197771]

pl.plot(p_len, f1, 'plum', label='f1', linestyle='-', marker='.')  # use pylab to plot x and y : Give your plots names
pl.plot(p_len, prec, 'cornflowerblue', label='precision', linestyle='-.', marker='.')
pl.plot(p_len, reca, 'lightskyblue', label='recall', linestyle='--', marker='.')

pl.xlabel('Prefixes length for BC5CDR-chem')  # make axis labels
# pl.ylabel('Performance')
# pl.legend()

# pl.legend()
plt.subplot(2,3,4)
# pl.plot("std=0.5")  # use pylab to plot x and y : Give your plots names
ncbi_data ='../read_data/data/Data_set/NCBI-disease/train.tsv'
ncbi=read_data(ncbi_data)
ncbi_mean = statistics.mean(ncbi)
ncbi_std = statistics.stdev(ncbi)
print('ncbi_mean',ncbi_mean)
print('ncbi_std',ncbi_std)

count_dict = dict()
for i in ncbi:
    if i in count_dict:
        count_dict[i]+=1
    else:
        count_dict[i]=1
print(count_dict)

ncbi_x = []
ncbi_y =[]
for item in count_dict.items():
    ncbi_x.append(item[0])
    ncbi_y.append(item[1])
plt.bar(ncbi_x,ncbi_y)
print(ncbi_mean)

pl.xlim(0,60)
pl.ylim(15, 240)
pl.ylabel('Frequency')  # make axis labels
pl.xlabel("NCBI-Disease context's length")
pl.text(42, 220, 'mean: 25.02')
pl.text(42, 205, 'std : 12.52')


# #######
plt.subplot(2,3,5)
bc2gm_data ='../read_data/data/Data_set/BC2GM/train.tsv'
bc2gm=read_data(bc2gm_data)

bc2gm_mean = statistics.mean(bc2gm)
bc2gm_std = statistics.stdev(bc2gm)
print('bc2gm_mean',bc2gm_mean)
print('bc2gm_std',bc2gm_std)

count_dict1 = dict()
for i in bc2gm:
    if i in count_dict1:
        count_dict1[i]+=1
    else:
        count_dict1[i]=1
print(count_dict1)

bc2gm_x = []
bc2gm_y =[]
for item1 in count_dict1.items():
    bc2gm_x.append(item1[0])
    bc2gm_y.append(item1[1])
plt.bar(bc2gm_x,bc2gm_y)

pl.xlim(0,70)
pl.ylim(15, 460)
# pl.ylabel('frequency')  # make axis labels
pl.xlabel("BC2GM context's length")
pl.text(48, 420, 'mean: 28.24')
pl.text(48, 390, 'std : 15.89')

# #####
plt.subplot(2,3,6)
BC5CDR_chem_data ='../read_data/data/Data_set/BC5CDR_chem/train.tsv'
BC5CDR_chem=read_data(BC5CDR_chem_data)

bc5chem_mean = statistics.mean(BC5CDR_chem)
bc5chem_std = statistics.stdev(BC5CDR_chem)

print('bc5chem_mean',bc5chem_mean)
print('bc5chem_std',bc5chem_std)

count_dict2 = dict()
for i in BC5CDR_chem:
    if i in count_dict2:
        count_dict2[i]+=1
    else:
        count_dict2[i]=1
print(count_dict2)

bc5chem_x = []
bc5chem_y =[]
for item2 in count_dict2.items():
    bc5chem_x.append(item2[0])
    bc5chem_y.append(item2[1])
plt.bar(bc5chem_x,bc5chem_y)

pl.xlim(0,65)
pl.ylim(15, 350)
# pl.ylabel('frequency')  # make axis labels
pl.xlabel("BC5CDR_chem context's length")
pl.text(44, 320, 'mean: 25.74')
pl.text(44, 298, 'std : 15.02')


# plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
plt.suptitle('relationship between context length and prefix length.')  # give plot a title
plt.show()