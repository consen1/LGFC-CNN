import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import gc
import argparse
import pandas as pd
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,Flatten,Dense,Concatenate,BatchNormalization,MaxPool2D,Dropout, Activation, TimeDistributed, Bidirectional,LSTM
from keras import Model
from keras import optimizers
from keras.layers.recurrent import GRU
import keras.backend.tensorflow_backend as KTF
import tensorflow.compat.v1 as tf
from keras.utils import np_utils
from sklearn import metrics
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model

def preprocess_data(X, scaler=None, stand = True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X

def calculate_performace(pred_res,pred_label, test_label):
    tn, fp, fn, tp = metrics.confusion_matrix(test_label, pred_label).ravel()
    auc = metrics.roc_auc_score(y_score=pred_res, y_true=test_label)
    ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)
    acc = metrics.accuracy_score(y_pred=pred_label,y_true=test_label)
    mcc = metrics.matthews_corrcoef(y_pred=pred_label,y_true=test_label)
    f1_score = metrics.f1_score(y_pred=pred_label,y_true=test_label)
    sensitive = tp/(tp+fn)
    specificity = tn/(tn+fp)
    ppv = tp/(tp+fp)

    print()
    print('test result')
    print('acc', acc)
    print('auc', auc)
    print('mcc', mcc)
    print('f1-score', f1_score)
    print('sensitive', sensitive)
    print('specificity', specificity)
    print('ppv', ppv)
    print('ap', ap)
    return acc, auc, mcc, f1_score, sensitive, specificity, ppv, ap

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array

def get_pro_seq_concolutional_array(seq, motif_len=4):
    alpha = 'AIYHRDC'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 7))
    for i in range(motif_len - 1):
        new_array[i] = np.array([1/7] * 7)

    for i in range(row - 3, row):
        new_array[i] = np.array([1/7] * 7)
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'AIYHRDC':
            new_array[i] = np.array([1/7] * 7)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        # data[key] = new_array
    return new_array

def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def split_overlap_seq(seq, window_size=101):
    overlap_size = 20
    # pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) // (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size) % (window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, max_len=window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            # pdb.set_trace()
            # start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len=window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs



def get_lncbag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    for index,seq in enumerate(seqs):
        # pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        # flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)

        if num_of_ins > channel:
            start = (num_of_ins - channel) // 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                # bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags
def get_probag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    for index,seq in enumerate(seqs):
        # pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        # flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_pro_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)

        if num_of_ins > channel:
            start = (num_of_ins - channel) // 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                # bag_subt.append(random.choice(bag_subt))
                tri_fea = get_pro_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags

def read_lncseq_graphprot(seq_file):
    seq_list = []
    name = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name.append(line[1:-1])
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
    return seq_list,name

def read_proseq_graphprot(seq_file):
    seq_list = []
    name = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name.append(line[1:-1])
            else:
                seq = line[:-1]
                seq_list.append(seq)
    return seq_list,name

def read_lncdata_file(dataall, train = True):
    data = dict()
    seqs,name = read_lncseq_graphprot(dataall)
    data["seq"] = seqs
    return data,name

def read_prodata_file(dataall, train = True):
    data = dict()
    seqs,name = read_proseq_graphprot(dataall)
    data["seq"] = seqs
    return data,name
def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq
def get_bag_lncdata_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        # flat_array = []
        bag_subt = []
        # for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags
def get_bag_prodata_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        # flat_array = []
        bag_subt = []
        # for bag_seq in bag_seqs:
        tri_fea = get_pro_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags
def get_lncdata(data, channel = 7,  window_size = 101, train = True):
    data,name = read_lncdata_file(data, train = train)
    if channel == 1:
        train_bags = get_bag_lncdata_1_channel(data, max_len = window_size)

    else:
        train_bags = get_lncbag_data(data, channel = channel, window_size = window_size)

    return train_bags,name
def get_prodata(data, channel = 7,  window_size = 101, train = True):
    data,name = read_prodata_file(data, train = train)
    if channel == 1:
        train_bags = get_bag_prodata_1_channel(data, max_len = window_size)

    else:
        train_bags = get_probag_data(data, channel = channel, window_size = window_size)

    return train_bags,name
def get_maclncdata(lncfea,dataset):
    lncDic = {}
    for index, i in enumerate(lncfea):
        if i.startswith('\n'):
            continue
        L = i[:-1].split('\t')
        lncDic[L[0]] = L[1:]
    return lncDic
def get_macprodata(profea,dataset):
    proDic = {}
    for index, i in enumerate(profea):
        L = i[:-1].split('\t')
        proDic[L[0]] = L[1:]
    return proDic

# def CNNBiGRU():
#
#     # Model
#     inx1 = Input(shape=(7, 4, 186))
#     inx2 = Input(shape=(7, 7, 96))
#     # inx3 = Input(shape=())
#     #
#     # Convolution layer
#     x1 = Conv2D(filters=16, kernel_size=(4, 10),padding='same', data_format='channels_first')(inx1)
#     x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
#     x1 = Activation('relu')(x1)
#     x1 = MaxPool2D((1,3),data_format='channels_first')(x1)
#     x1 = Conv2D(filters=32, kernel_size=(4,10), padding='same', data_format='channels_first')(x1)
#     x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
#     x1 = Activation('relu')(x1)
#     x1 = GlobalMaxPooling2D(data_format='channels_first')(x1)
#     x1 = Dropout(0.2)(x1)
#     x2 = Conv2D(filters=16, kernel_size=(7, 10), padding='same', data_format='channels_first')(inx2)
#     x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
#     x2 = Activation('relu')(x2)
#     x2 = MaxPool2D(pool_size=(1,3),data_format='channels_first')(x2)
#     x2 = Conv2D(filters=32, kernel_size=(7, 10), padding='same', data_format='channels_first')(x2)
#     x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
#     x2 = Activation('relu')(x2)
#     x2 = GlobalMaxPooling2D(data_format='channels_first')(x2)
#     x2 = Dropout(0.2)(x2)
#     #Concatenate
#     x = Concatenate()([x1, x2])
#     x = Dense(32)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # fully-connected layer
#     x = Dense(2)(x)
#     xout = Activation('softmax')(x)
#
#     model = Model(inputs=[inx1, inx2], outputs=[xout])
#     print(model.summary())
#     return model

def Prediction(dataname):
    # config GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    if dataname == '7317' or 'ran7317':
        lnc = './Datasets/Train_dataset/NPinter_human/RNA_human_fasta.fasta'
        pro = './Datasets/Train_dataset/NPinter_human/protein_human_fasta.fasta'
        lncFe = open('Datasets/Train_dataset/NPinter_human/lncRED.fasta', 'r').readlines()
        lncFe4 = open('./Datasets/Train_dataset/NPinter_human/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('Datasets/Train_dataset/NPinter_human/lnc3mer.fasta', 'r').readlines()
        proFe = open('Datasets/Train_dataset/NPinter_human/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/Train_dataset/NPinter_human/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/Train_dataset/NPinter_human/pro4mer.fasta', 'r').readlines()
    elif dataname == '1847' or 'ran1847':
        lnc = './Datasets/Train_dataset/NPinter_mouse/RNA_mouse_fasta.fasta'
        pro = './Datasets/Train_dataset/NPinter_mouse/protein_mouse_fasta.fasta'
        lncFe = open('Datasets/Train_dataset/NPinter_mouse/lncRED.fasta', 'r').readlines()
        lncFe4 = open('./Datasets/Train_dataset/NPinter_mouse/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('Datasets/Train_dataset/NPinter_mouse/lnc3mer.fasta', 'r').readlines()
        proFe = open('Datasets/Train_dataset/NPinter_mouse/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/Train_dataset/NPinter_mouse/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/Train_dataset/NPinter_mouse/pro4mer.fasta', 'r').readlines()
    elif dataname == '21850' or 'ran21850':
        lnc = './Datasets/Train_dataset/lncseq.fasta'
        pro = './Datasets/Train_dataset/proseq.fasta'
        lncFe = open('Datasets/machineFea/lncRED.fasta', 'r').readlines()
        lncFe4 = open('Datasets/machineFea/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('./Datasets/machineFea/lnckmer3.fasta', 'r').readlines()
        proFe = open('Datasets/machineFea/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/machineFea/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/machineFea/pro4mer.fasta', 'r').readlines()
    findlnclen = open(lnc, 'r').readlines()
    findprolen = open(pro, 'r').readlines()
    lnclen = 0
    lncnum = 0
    for i in findlnclen:
        if i.startswith('>'):
            continue
        lncnum += 1
        lnclen += len(i) - 1
    prolen = 0
    pronum = 0
    for i in findprolen:
        if i.startswith('>'):
            continue
        pronum += 1
        prolen += len(i) - 1
    print(lnclen)
    print(prolen)
    print(lnclen // lncnum)
    print(prolen // pronum)
    lnc_channel = 7
    pro_channel = 7
    lnc_window_size = lnclen // lncnum // 7
    pro_window_size = prolen // pronum // 7
    lnc_glochannel = 1
    pro_glochannel = 1
    lnc_glowindow = lnclen // lncnum
    pro_glowindow = prolen // pronum
    train_lnc, lnc_name = get_lncdata(lnc, channel=lnc_channel, window_size=lnc_window_size)
    train_pro, pro_name = get_prodata(pro, channel=pro_channel, window_size=pro_window_size)
    train_glolnc, lnc_gloname = get_lncdata(lnc, channel=lnc_glochannel, window_size=lnc_glowindow)
    train_glopro, pro_gloname = get_prodata(pro, channel=pro_glochannel, window_size=pro_glowindow)
    print(lnc_window_size, pro_window_size, lnc_glowindow, pro_glowindow)

    lncDic = {name: seq for name, seq in zip(lnc_name, train_lnc)}
    proDic = {name: seq for name, seq in zip(pro_name, train_pro)}
    glolncDic = {name: seq for name, seq in zip(lnc_name, train_glolnc)}
    gloproDic = {name: seq for name, seq in zip(pro_name, train_glopro)}
    if dataname == '21850':
        data = open('Datasets/Train_dataset/RPI21850.txt', 'r').readlines()
        seed = 29
    elif dataname == 'ran21850':
        data = open('Datasets/Train_dataset/ran21850.txt', 'r').readlines()
        seed = 29
    elif dataname == '7317':
        data = open('Datasets/Train_dataset/NPinter_human/RPI7317.txt', 'r').readlines()
        seed = 8
    elif dataname == 'ran7317':
        data = open('Datasets/Train_dataset/NPinter_human/ran7317.txt', 'r').readlines()
        seed = 8
    elif dataname == '1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/RPI1847.txt', 'r').readlines()
        seed = 13
    elif dataname == 'ran1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/ran1847.txt', 'r').readlines()
        seed = 13
    out = open('./out/' + dataname + '.txt', 'w')
    data_Lst = np.array([i.split() for i in data])
    dataset = data_Lst[:, 0:2]
    labels = data_Lst[:, 2]
    y, encoder = preprocess_labels(labels)
    # CNNdata
    X_train, X_test_a, Y_train, Y_test_a = train_test_split(dataset, y, test_size=0.3, stratify=y,
                                                            random_state=seed)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                    random_state=seed)
    del X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))
    # macdata
    dataall1 = get_maclncdata(lncFe, dataset)
    dataall2 = get_maclncdata(lncFe4, dataset)
    dataall3 = get_maclncdata(lncFe5, dataset)
    dataall4 = get_macprodata(proFe10, dataset)
    dataall5 = get_macprodata(proFe, dataset)
    dataall6 = get_macprodata(proFe9, dataset)
    datalncall = [dataall1[i[0]] + dataall2[i[0]] + dataall3[i[0]] for i in dataset]
    datalncall = preprocess_data(datalncall)
    maclncdata = np.array(datalncall)
    dataproall = [dataall4[i[1]] + dataall5[i[1]] + dataall6[i[1]] for i in dataset]
    dataproall = preprocess_data(dataproall)
    macprodata = np.array(dataproall)
    X_train1, X_test_a, Y_train, Y_test_a = train_test_split(maclncdata, y, test_size=0.3, stratify=y,
                                                             random_state=seed)
    X_test1, X_val1, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                      random_state=seed)
    del X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train1), len(X_test1), len(X_val1)))
    X_train2, X_test_a, Y_train, Y_test_a = train_test_split(macprodata, y, test_size=0.3, stratify=y,
                                                             random_state=seed)
    X_test2, X_val2, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                      random_state=seed)
    del dataset, X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train2), len(X_test2), len(X_val2)))
    # localdata
    test_lnc_data = np.array([lncDic[i[0]] for i in X_test if i[0] in lncDic])
    test_pro_data = np.array([proDic[i[1]] for i in X_test if i[1] in proDic])
    # globaldata

    test_glolnc_data = np.array([glolncDic[i[0]] for i in X_test if i[0] in glolncDic])
    test_glopro_data = np.array([gloproDic[i[1]] for i in X_test if i[1] in gloproDic])
    # macdata
    test_mac1_data = X_test1
    test_mac2_data = X_test2
    real_labels = []
    for val in Y_test:
        if val[0] == 1:
            real_labels.append(0)
        else:
            real_labels.append(1)

    model1 = load_model("./Models/"+dataname+".h5")

    testres1 = model1.predict([test_lnc_data, test_pro_data,test_glolnc_data,test_glopro_data,test_mac1_data,test_mac2_data], verbose=1)

    pred_res1 = testres1[:, 1]
    proba_res1 = transfer_label_from_prob(pred_res1)
    test_label1 = [int(x) for x in real_labels]
    acc, auc, mcc, f1_score, sensitive, specificity, ppv, ap =calculate_performace(pred_res1, proba_res1, test_label1)
    out.write('Mine\t')
    performace = [acc, auc, mcc, f1_score, sensitive, specificity, ppv, ap]
    for i in performace:
        out.write(str(i) + '\t')
    out.write('\n')

    model_name = "Mine"+dataname
    fpr, tpr, thresholds = metrics.roc_curve(y_score=pred_res1, y_true=test_label1)
    precision, recall, thresholds = metrics.precision_recall_curve(probas_pred=pred_res1, y_true=test_label1)

    auc_dict = {"fpr": fpr, "tpr": tpr}
    ap_dict = {"precision": precision, "recall": recall}

    auc_df = pd.DataFrame(data=auc_dict, index=None)
    ap_df = pd.DataFrame(data=ap_dict, index=None)

    auc_df.to_csv("./out/" + model_name + "_auc.csv", index=None)
    ap_df.to_csv("./out/" + model_name + "_ap.csv", index=None)
    np.savez("./result/" + model_name + "_reslut", pred_res1, proba_res1, test_label1)

parser = argparse.ArgumentParser(description="LGFC-CNN：Prediction of lncRNA-protein interactions by using multiple types of features through deep learning")
parser.add_argument('-dataset', type=str, help='RPI21850,RPI7317 or RPI1847')
args = parser.parse_args()
dataname = args.dataset
Prediction(dataname)