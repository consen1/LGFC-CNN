import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import gc
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,Flatten,Dense,Concatenate,BatchNormalization,MaxPool2D,Dropout, Activation, TimeDistributed, Bidirectional,LSTM
from keras import Model
from keras import optimizers
from keras.layers.recurrent import GRU
import keras.backend.tensorflow_backend as KTF
import tensorflow.compat.v1 as tf
from keras.utils import np_utils
from sklearn import metrics
from keras import regularizers
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def LGFC_CNN(lnc_window_size,pro_window_size,lnc_glowindow,pro_glowindow,
                                    train_lnc_data,train_pro_data,train_glolnc_data,train_glopro_data,train_mac1_data,train_mac2_data,Y_train,
                                    val_lnc_data,val_pro_data,val_glolnc_data,val_glopro_data,val_mac1_data,val_mac2_data,Y_val,
                                    test_lnc_data,test_pro_data,test_glolnc_data,test_glopro_data,test_mac1_data,test_mac2_data,Y_test,real_labels):

    # local
    inx1 = Input(shape=(7, 4, lnc_window_size+6))
    inx2 = Input(shape=(7, 7, pro_window_size+6))
    filter1 = 16
    filter2 = 32
    kernel_size1 = (4,40)
    kernel_size2 = (7,40)
    dense1 = 32
    # global
    inx3 = Input(shape=(1, 4, lnc_glowindow+6))
    inx4 = Input(shape=(1, 7, pro_glowindow+6))
    filter3 = 32
    filter4 = 64
    kernel_size3 = (4,30)
    kernel_size4 = (7,30)
    dense3 = 64
    # FC
    inx5 = Input(shape=(117,))
    inx6 = Input(shape=(2764,))
    # local
    # Convolution layer
    x1 = Conv2D(filters=filter1, kernel_size=kernel_size1,strides=1,padding='same', data_format='channels_first')(inx1)
    x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPool2D(pool_size=4,strides=4,data_format='channels_first')(x1)
    x1 = Conv2D(filters=filter2, kernel_size=kernel_size1,strides=1, padding='same', data_format='channels_first')(x1)
    x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
    x1 = Activation('relu')(x1)
    x1 = GlobalMaxPooling2D(data_format='channels_first')(x1)
    x1 = Dropout(0.2)(x1)
    x2 = Conv2D(filters=filter1, kernel_size=kernel_size2,strides=1, padding='same', data_format='channels_first')(inx2)
    x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPool2D(pool_size=7,strides=7,data_format='channels_first')(x2)
    x2 = Conv2D(filters=filter2, kernel_size=kernel_size2,strides=1, padding='same', data_format='channels_first')(x2)
    x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
    x2 = Activation('relu')(x2)
    x2 = GlobalMaxPooling2D(data_format='channels_first')(x2)
    x2 = Dropout(0.2)(x2)
    # Concatenate
    xlocal = Concatenate()([x1, x2])
    xlocal = Dense(dense1)(xlocal)
    # global
    # Convolution layer
    x3 = Conv2D(filters=filter3, kernel_size=kernel_size3,strides=1,padding='same', data_format='channels_first')(inx3)
    x3 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPool2D(pool_size=4,strides=4,data_format='channels_first')(x3)
    x3 = Conv2D(filters=filter4, kernel_size=kernel_size3, strides=1,padding='same', data_format='channels_first')(x3)
    x3 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = GlobalMaxPooling2D(data_format='channels_first')(x3)
    x3 = Dropout(0.2)(x3)
    x4 = Conv2D(filters=filter3, kernel_size=kernel_size4,strides=1, padding='same', data_format='channels_first')(inx4)
    x4 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPool2D(pool_size=7,strides=7,data_format='channels_first')(x4)
    x4 = Conv2D(filters=filter4, kernel_size=kernel_size4,strides=1, padding='same', data_format='channels_first')(x4)
    x4 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x4)
    x4 = Activation('relu')(x4)
    x4 = GlobalMaxPooling2D(data_format='channels_first')(x4)
    x4 = Dropout(0.2)(x4)
    # Concatenate
    xglobal = Concatenate()([x3, x4])
    xglobal = Dense(dense3)(xglobal)
    xmac1 = Dense(64, activation='relu')(inx5)
    xmac1 = Dense(32, activation='relu')(xmac1)
    xmac2 = Dense(512, activation='relu')(inx6)
    xmac2 = Dense(128, activation='relu')(xmac2)
    xmac = Concatenate()([xmac1, xmac2])
    xmac = Dense(32)(xmac)
    x = Concatenate()([xlocal,xglobal,xmac])
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # fully-connected layer
    x = Dense(2)(x)
    xout = Activation('softmax')(x)

    model = Model(inputs=[inx1,inx2,inx3,inx4,inx5,inx6], outputs=[xout])
    print(model.summary())
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Training --------------')
    model.fit(x=[train_lnc_data, train_pro_data,train_glolnc_data,train_glopro_data,train_mac1_data,train_mac2_data], y=Y_train, validation_data=([val_lnc_data, val_pro_data,val_glolnc_data,val_glopro_data,val_mac1_data,val_mac2_data], Y_val),
              batch_size=128, epochs=45, shuffle=True, verbose=1, callbacks=[])
    # test
    print('\nTesting---------------')
    loss, accuracy = model.evaluate([test_lnc_data, test_pro_data,test_glolnc_data,test_glopro_data,test_mac1_data,test_mac2_data], Y_test, verbose=1)
    print(loss, accuracy)

    # get the confidence probability
    testres = model.predict([test_lnc_data, test_pro_data,test_glolnc_data,test_glopro_data,test_mac1_data,test_mac2_data], verbose=1)
    pred_res = testres[:, 1]
    proba_res = transfer_label_from_prob(pred_res)
    test_label = [int(x) for x in real_labels]
    calculate_performace(pred_res, proba_res, test_label)
    return accuracy,model

def newwaycon(dataname):
    # config GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    if dataname=='7317' or 'ran7317':
        #sequences
        lnc = './Datasets/Train_dataset/NPinter_human/RNA_human_fasta.fasta'
        pro = './Datasets/Train_dataset/NPinter_human/protein_human_fasta.fasta'
        # hand-designed features
        lncFe = open('Datasets/Train_dataset/NPinter_human/lncRED.fasta', 'r').readlines()
        lncFe4 = open('./Datasets/Train_dataset/NPinter_human/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('Datasets/Train_dataset/NPinter_human/lnc3mer.fasta', 'r').readlines()
        proFe = open('Datasets/Train_dataset/NPinter_human/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/Train_dataset/NPinter_human/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/Train_dataset/NPinter_human/pro4mer.fasta', 'r').readlines()
    elif dataname=='1847' or 'ran1847':
        lnc = './Datasets/Train_dataset/NPinter_mouse/RNA_mouse_fasta.fasta'
        pro = './Datasets/Train_dataset/NPinter_mouse/protein_mouse_fasta.fasta'
        lncFe = open('Datasets/Train_dataset/NPinter_mouse/lncRED.fasta', 'r').readlines()
        lncFe4 = open('./Datasets/Train_dataset/NPinter_mouse/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('Datasets/Train_dataset/NPinter_mouse/lnc3mer.fasta', 'r').readlines()
        proFe = open('Datasets/Train_dataset/NPinter_mouse/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/Train_dataset/NPinter_mouse/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/Train_dataset/NPinter_mouse/pro4mer.fasta', 'r').readlines()
    elif dataname=='21850' or 'ran21850':
        lnc = './Datasets/Train_dataset/lncseq.fasta'
        pro = './Datasets/Train_dataset/proseq.fasta'
        lncFe = open('Datasets/machineFea/lncRED.fasta', 'r').readlines()
        lncFe4 = open('Datasets/machineFea/lncDNC.fasta', 'r').readlines()
        lncFe5 = open('./Datasets/machineFea/lnckmer3.fasta', 'r').readlines()
        proFe = open('Datasets/machineFea/proAAC.fasta', 'r').readlines()
        proFe9 = open('Datasets/machineFea/pro3mer.fasta', 'r').readlines()
        proFe10 = open('Datasets/machineFea/pro4mer.fasta', 'r').readlines()
    findlnclen = open(lnc,'r').readlines()
    findprolen = open(pro,'r').readlines()
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
    print(lnclen//lncnum)
    print(prolen//pronum)
    lnc_channel = 7
    pro_channel = 7
    lnc_window_size = lnclen // lncnum // 7
    pro_window_size = prolen // pronum // 7
    lnc_glochannel = 1
    pro_glochannel = 1
    lnc_glowindow = lnclen // lncnum
    pro_glowindow = prolen // pronum
    train_lnc,lnc_name = get_lncdata(lnc, channel=lnc_channel, window_size=lnc_window_size)
    train_pro,pro_name = get_prodata(pro, channel=pro_channel, window_size=pro_window_size)
    train_glolnc, lnc_gloname = get_lncdata(lnc, channel=lnc_glochannel, window_size=lnc_glowindow)
    train_glopro, pro_gloname = get_prodata(pro, channel=pro_glochannel, window_size=pro_glowindow)
    print(lnc_window_size,pro_window_size,lnc_glowindow,pro_glowindow)


    lncDic = {name:seq for name,seq in zip(lnc_name,train_lnc)}
    proDic = {name:seq for name,seq in zip(pro_name,train_pro)}
    glolncDic = {name:seq for name,seq in zip(lnc_name,train_glolnc)}
    gloproDic = {name:seq for name,seq in zip(pro_name,train_glopro)}
    if dataname == '21850':
        data = open('Datasets/Train_dataset/RPI21850.txt', 'r').readlines()
        seed = 29
    elif dataname == 'ran21850':
        data = open('Datasets/Train_dataset/ran21850.txt', 'r').readlines()
        seed = 29
    elif dataname=='7317':
        data = open('Datasets/Train_dataset/NPinter_human/RPI7317.txt', 'r').readlines()
        seed = 8
    elif dataname=='ran7317':
        data = open('Datasets/Train_dataset/NPinter_human/ran7317.txt', 'r').readlines()
        seed = 8
    elif dataname=='1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/RPI1847.txt', 'r').readlines()
        seed = 13
    elif dataname=='ran1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/ran1847.txt', 'r').readlines()
        seed = 13
    data_Lst = np.array([i.split() for i in data])
    dataset = data_Lst[:, 0:2]
    labels = data_Lst[:, 2]
    y, encoder = preprocess_labels(labels)
    #CNNdata
    X_train, X_test_a, Y_train, Y_test_a = train_test_split(dataset, y, test_size=0.3, stratify=y,
                                                            random_state=seed)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                    random_state=seed)
    del X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))
    #macdata
    dataall1 = get_maclncdata(lncFe, dataset)
    dataall2 = get_maclncdata(lncFe4,dataset)
    dataall3 = get_maclncdata(lncFe5,dataset)
    dataall4 = get_macprodata(proFe10, dataset)
    dataall5 = get_macprodata(proFe,dataset)
    dataall6 = get_macprodata(proFe9,dataset)
    datalncall = [dataall1[i[0]]+dataall2[i[0]]+dataall3[i[0]] for i in dataset]
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
    #localdata
    train_lnc_data = np.array([lncDic[i[0]] for i in X_train if i[0] in lncDic])
    train_pro_data = np.array([proDic[i[1]] for i in X_train if i[1] in proDic])
    val_lnc_data = np.array([lncDic[i[0]] for i in X_val if i[0] in lncDic])
    val_pro_data = np.array([proDic[i[1]] for i in X_val if i[1] in proDic])
    test_lnc_data = np.array([lncDic[i[0]] for i in X_test if i[0] in lncDic])
    test_pro_data = np.array([proDic[i[1]] for i in X_test if i[1] in proDic])
    #globaldata
    train_glolnc_data = np.array([glolncDic[i[0]] for i in X_train if i[0] in glolncDic])
    train_glopro_data = np.array([gloproDic[i[1]] for i in X_train if i[1] in gloproDic])
    val_glolnc_data = np.array([glolncDic[i[0]] for i in X_val if i[0] in glolncDic])
    val_glopro_data = np.array([gloproDic[i[1]] for i in X_val if i[1] in gloproDic])
    test_glolnc_data = np.array([glolncDic[i[0]] for i in X_test if i[0] in glolncDic])
    test_glopro_data = np.array([gloproDic[i[1]] for i in X_test if i[1] in gloproDic])
    #macdata
    train_mac1_data = X_train1
    val_mac1_data = X_val1
    test_mac1_data = X_test1
    train_mac2_data = X_train2
    val_mac2_data = X_val2
    test_mac2_data = X_test2
    real_labels = []
    for val in Y_test:
        if val[0] == 1:
            real_labels.append(0)
        else:
            real_labels.append(1)
    val_label_new = []
    for val in Y_val:
        if val[0] == 1:
            val_label_new.append(0)
        else:
            val_label_new.append(1)
    train_label_new = []
    for val in Y_train:
        if val[0] == 1:
            train_label_new.append(0)
        else:
            train_label_new.append(1)


    accuracy1,model1 = LGFC_CNN(lnc_window_size,pro_window_size,lnc_glowindow,pro_glowindow,
                                train_lnc_data,train_pro_data,train_glolnc_data,train_glopro_data,train_mac1_data,train_mac2_data,Y_train,
                                val_lnc_data,val_pro_data,val_glolnc_data,val_glopro_data,val_mac1_data,val_mac2_data,Y_val,
                                test_lnc_data,test_pro_data,test_glolnc_data,test_glopro_data,test_mac1_data,test_mac2_data,Y_test,real_labels)
    model1.save("./Models/"+dataname+".h5")
parser = argparse.ArgumentParser(description="LGFC-CNN：Prediction of lncRNA-protein interactions by using multiple types of features through deep learning")
parser.add_argument('-dataset', type=str, help='RPI21850,RPI7317 or RPI1847')
args = parser.parse_args()
datname = args.dataset
newwaycon(datname)
