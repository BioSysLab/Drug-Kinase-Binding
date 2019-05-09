from __future__ import division, print_function
from comet_ml import Experiment
import numpy as np
from numpy import inf, ndarray
import pandas as pd
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
import keras.backend as K
from keras.models import model_from_json
from keras.models import load_model
from tempfile import TemporaryFile
from keras import layers
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
#matplotlib inline
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import CuDNNLSTM, Dense, Bidirectional
from keras.regularizers import l2
from functools import partial
from multiprocessing import cpu_count, Pool
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from NGF.utils import filter_func_args, mol_shapes_to_dims
import NGF.utils
import NGF_layers.features
import NGF_layers.graph_layers
from NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features
from NGF_layers.features import padaxis, tensorise_smiles, concat_mol_tensors
from NGF_layers.graph_layers import temporal_padding, neighbour_lookup, NeuralGraphHidden, NeuralGraphOutput

tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

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

XT_train, Y_train, Y_digital_train = make_datasets(df)

smiles_train = df['canonical_smiles']
X_atoms_train, X_bonds_train, X_edges_train = tensorise_smiles(smiles_train, max_degree=5, max_atoms = 66)
num_molecules = X_atoms_train.shape[0]
max_atoms = X_atoms_train.shape[1]
max_degree = X_bonds_train.shape[2]
num_atom_features = X_atoms_train.shape[-1]
num_bond_features = X_bonds_train.shape[-1]

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
     'dropout': 0.25,
     'l2reg': 0.05}


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def get_cindex(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)

def dtb_model_2(params,lr_value, windows_seq):

    atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features),dtype = 'float32')
    bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features),dtype = 'float32')
    edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
    XTinput = keras.layers.Input(shape = (1215, dict_prot_len))

    g1 = NeuralGraphHidden(params['conv_width'] , activ = None, bias = True , init = 'glorot_uniform')([atoms0,bonds,edges])
    g1 = BatchNormalization(momentum=0.6)(g1)
    g1 = Activation('relu')(g1)
    g2 = NeuralGraphHidden(params['conv_width'] , activ = None, bias = True , init = 'glorot_uniform')([g1,bonds,edges])
    g2 = BatchNormalization(momentum=0.6)(g2)
    g2 = Activation('relu')(g2)

    fp_out0 = NeuralGraphOutput(params['fp_length'], activ=None, bias = False , init = 'glorot_uniform')([atoms0, bonds, edges])
    fp_out0 = BatchNormalization(momentum=0.6)(fp_out0)
    fp_out0 = Activation('softmax')(fp_out0)
    fp_out1 = NeuralGraphOutput(params['fp_length'], activ=None, bias = False , init = 'glorot_uniform')([g1, bonds, edges])
    fp_out1 = BatchNormalization(momentum=0.6)(fp_out1)
    fp_out1 = Activation('softmax')(fp_out1)
    fp_out2 = NeuralGraphOutput(params['fp_length'], activ=None, bias = False , init = 'glorot_uniform')([g2, bonds, edges])
    fp_out2 = BatchNormalization(momentum=0.6)(fp_out2)
    fp_out2 = Activation('softmax')(fp_out2)

    encode_smiles = keras.layers.add([fp_out0,fp_out1,fp_out2])

    encode_smiles = BatchNormalization(momentum=0.6)(encode_smiles)

    encode_protein = keras.layers.Conv1D(filters = windows_seq, kernel_size = params['size_protein_1'], activation= None, padding = 'same', strides = 1)(XTinput)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)
    encode_protein = keras.layers.Conv1D(filters = 2*windows_seq, kernel_size = params['size_protein_1'], activation = None, padding = 'same', strides = 1)(encode_protein)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)
    encode_protein = keras.layers.Conv1D(filters = 3*windows_seq, kernel_size = params['size_protein_1'], activation = None, padding = 'same', strides = 1)(encode_protein)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)
    encode_protein = Activation('relu')(encode_protein)

    encode_protein = keras.layers.GlobalMaxPooling1D()(encode_protein)
    encode_protein = BatchNormalization(momentum=0.6)(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])


    fc1 = keras.layers.Dense(params['dense_size'],activation = None,kernel_regularizer=regularizers.l2(params['l2reg']))(encode_interaction)
    fc1 = BatchNormalization(momentum=0.6)(fc1)
    fc1 = Activation('relu')(fc1)
    fc2 = keras.layers.Dropout(0.25)(fc1)
    fc2 = keras.layers.Dense(params['dense_size'],activation = None,kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)
    fc2 = BatchNormalization(momentum=0.6)(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = keras.layers.Dropout(0.25)(fc2)
    #fc3 = keras.layers.Dense(params['dense_size'],activation = None,kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)
    #fc3 = BatchNormalization(momentum=0.6)(fc3)
    #fc3 = Activation('relu')(fc3)


    predictions = keras.layers.Dense(1, kernel_initializer='normal')(fc2)

    interactionModel = keras.Model(inputs=[atoms0, bonds, edges, XTinput], outputs=[predictions])

    print(interactionModel.summary())
    adam = keras.optimizers.Adam(lr=lr_value, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    interactionModel.compile(
        optimizer= adam,
        loss='mean_squared_error',
        metrics=['mse', get_cindex, r_square]
    )
    return interactionModel

graph_model1 = dtb_model_2(p,0.001,32)
h5 = History()
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=2, min_lr=0.00001, verbose=1, min_delta=1e-5)
graph_model1.fit([X_atoms_train,X_bonds_train,X_edges_train,XT_train],Y_train,
                    batch_size = 128 , epochs = 30, validation_split=0.05,
                shuffle= True,callbacks= [h5, rlr])
