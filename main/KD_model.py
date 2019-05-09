import numpy as np
import pandas as pd
from comet_ml import Experiment
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.models import model_from_json
from keras.models import load_model
from tempfile import TemporaryFile
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
#%matplotlib inline
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import CuDNNLSTM, Dense, Bidirectional

tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

# read training data
df = pd.read_csv(os.path.join(input_dir,'training_final.csv'),index_col=0)


dict_prot = { "A": 1, "C": 2, "E": 3, "D": 4, "G": 5,
                "F": 6, "I": 7, "H": 8, "K": 9, "M": 10, "L": 11,
                "N": 12, "Q": 13, "P": 14, "S": 15, "R": 16,
                "T": 17, "W": 18,
                "V": 19, "Y": 20 ,"X" : 21}

dict_prot_len = len(dict_prot)

def one_hot_sequence(protein, max_prot_len = 1215, prot_dict = dict_prot):
    X = np.zeros((max_prot_len, len(prot_dict)))
    for i,ch in enumerate(protein[:max_prot_len]):
        X[i, (prot_dict[ch])-1] = 1
    return (X)


def make_datasets(dataframe):

    XT = []
    bins = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    for i in range(dataframe.shape[0]):
        target = one_hot_sequence(dataframe.sequences.iloc[i])
        XT.append(target)
    XT = np.array(XT)
    print('Kinases encoded and ready for input')

    #Y = np.log10(dataframe.KD)#-np.mean(np.log10(dataframe.KD))
    Y = dataframe.pkd
    Y_digital = np.digitize(Y, bins, right=True)
    #Y_class = dataframe.activity
    print('Affinities logged, normalized, and ready for input')
    return(XT, Y, Y_digital)

### Y_digital can be used for stratified kfold validation split (Kd converted to bins)
XT_train, Y_train, Y_digital_train = make_datasets(df)

### smiles dictionary
char_to_int2 = {'.': 0, 's': 1, 'C': 2, '4': 3, 'i': 4, ')': 5,'o': 6,'K': 7,'-': 8,'B': 9,'c': 10,']': 11,'O': 12,'N': 13,'P': 14, '@': 15,
 'L': 16, '\\': 17,'1': 18, 'e': 19, 'S': 20, 'a': 21, 'E': 22, 'r': 23, 'I': 24, '+': 25, 'n': 26, '#': 27, '5': 28, 'H': 29, 'l': 30, '!': 31, '/': 32, '(': 33, '6': 34, '7': 35, 'F': 36, '8': 37, '2': 38, '=': 39, '3': 40, '[': 41}

int_to_char2 = {0: '.',1: 's', 2: 'C', 3: '4', 4: 'i', 5: ')', 6: 'o', 7: 'K', 8: '-', 9: 'B', 10: 'c', 11: ']', 12: 'O', 13: 'N', 14: 'P', 15: '@',
 16: 'L', 17: '\\', 18: '1', 19: 'e', 20: 'S', 21: 'a', 22: 'E', 23: 'r', 24: 'I', 25: '+', 26: 'n', 27: '#', 28: '5', 29: 'H', 30: 'l', 31: '!', 32: '/', 33: '(', 34: '6', 35: '7', 36: 'F', 37: '8', 38: '2', 39: '=', 40: '3', 41: '['}

### set the max length of the smiles strings in the data + 2 (for start char and end char)
embed2 = 102

def vectorize2(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed2 , 42 ),dtype=np.int8)
        for i,smile in enumerate(smiles):
            #encode the startchar
            one_hot[i,0,char_to_int2["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int2[c]] = 1
            #Encode endchar
            one_hot[i,len(smile)+1:,char_to_int2["E"]] = 1
        #Return two, one for input and the other for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]

### dummy is X_train shifted by one letter (used for the training of the autoencoder)
XD_train, dummy = vectorize2(df.canonical_smiles.values)

### load the SMILES encoder
smile_encoder = load_model(os.path.join(input_dir,'test_encoder.h5'))
print(smile_encoder.summary())

### calculate the encodings of the training data and normalize them
encodings = smile_encoder.predict(XD_train)
ep = 0.000000001
encodings_norm = encodings / (encodings.max(axis=0) + ep)

### define custom metrics
def get_cindex(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

### define the parameter dictionary
p = {'lr': 0.001,
     'nfilters': int(32),
     'size': int(8),
     'conv_width' : 32,
     'fp_length' : 96,
     'size_drug_1' : 8,
     'size_drug_2' : 4,
     'size_protein_1' : 8,
     'size_protein_2' : 16,
     'size_protein_3' : 3,
     'batch_size': int(128),
     'dense_size': int(192),
     'dense_size_2': 512,
     'dropout': 0.15,
     'l2reg': 0.00}

### define the model
def dtb_cheat_conv(params, lr_value, windows_seq):
    XDinput = keras.layers.Input(shape = (128,))
    XTinput = keras.layers.Input(shape = (1215, dict_prot_len))


    encode_protein = keras.layers.Conv1D(filters = windows_seq, kernel_size = 12, activation= None, padding = 'same', strides = 1,use_bias=False)(XTinput)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)

    encode_protein = keras.layers.Conv1D(filters = 2*windows_seq, kernel_size = 4, activation = None, padding = 'same', strides = 1,use_bias=False)(encode_protein)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)

    encode_protein = keras.layers.Conv1D(filters = 4*windows_seq, kernel_size = 4, activation = None, padding = 'same', strides = 1,use_bias=False)(encode_protein)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)

    encode_protein = keras.layers.GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([XDinput, encode_protein])
    encode_interaction = BatchNormalization(momentum=0.6)(encode_interaction)


    fc1 = keras.layers.Dense(256,activation = None ,kernel_regularizer=regularizers.l2(params['l2reg']),kernel_initializer='glorot_uniform')(encode_interaction)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = keras.layers.Dropout(0.3)(fc1)
    fc2 = keras.layers.Dense(256,activation = None ,kernel_regularizer=regularizers.l2(params['l2reg']),kernel_initializer='glorot_uniform')(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = keras.layers.Dropout(0.3)(fc2)

    predictions = keras.layers.Dense(1, kernel_initializer='normal')(fc2)

    interactionModel = keras.Model(inputs=[XDinput, XTinput], outputs=[predictions])

    #print(interactionModel.summary())
    adam = keras.optimizers.Adam(lr=lr_value, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    interactionModel.compile(
        optimizer= adam,
        loss='mean_squared_error',
        metrics=['mse', get_cindex, r_square]
    )
    return interactionModel

### read the test set (used for the competition)
#test = pd.read_csv(os.path.join(input_dir,'test_all.csv'),index_col=0)
#XT_test, Y_test, Y_digital_test = make_datasets(test)
#XD_test, dummy = vectorize2(test.canonical_smiles.values)
#encodings_test = smile_encoder.predict(XD_test)
#encodings_test_norm = encodings_test / (encodings_test.max(axis=0) + ep)


cheat_conv = dtb_cheat_conv(p, 0.001, 32)

print(cheat_conv.summary())

h5 = History()
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=2, min_lr=0.00001, verbose=1, min_delta=1e-5)

experiment = Experiment(api_key="U2SVXKVZDCjLHA8Dtl2FqpGWf",
                        project_name="general", workspace="tukerjerbss")

cheat_conv.fit([encodings,XT_train],Y_train, batch_size = 128 , epochs = 22, callbacks= [h5, rlr], shuffle = True ,validation_split=0.05)

cheat_conv.save('KD_model.h5')
