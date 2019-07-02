#!/usr/bin/env python
# coding: utf-8


from __future__ import division, print_function
import time

import numpy as np
from numpy import inf, ndarray
import pandas as pd
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn import metrics
import re
import scipy

from comet_ml import Experiment
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras import optimizers
from keras import losses
from keras import regularizers
import keras.backend as K
from keras.models import load_model
from keras import layers
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import CuDNNLSTM, Dense, Bidirectional
from keras.regularizers import l2
from keras.utils.generic_utils import Progbar

from ..main.NGF.utils import filter_func_args, mol_shapes_to_dims
import ..main.NGF.utils
import ..main.NGF_layers.features
import ..main.NGF_layers.graph_layers
from ..main.NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features
from ..main.NGF_layers.features import padaxis, tensorise_smiles, concat_mol_tensors
from ..main.NGF_layers.graph_layers import temporal_padding, neighbour_lookup, NeuralGraphHidden, NeuralGraphOutput




tf.set_random_seed(1); np.random.seed(1); random.seed(1)


# ## Load model


max_atoms=66
max_degree=5
num_atom_features=62
num_bond_features=6



dict_prot = { "A": 1, "C": 2, "E": 3, "D": 4, "G": 5,
                "F": 6, "I": 7, "H": 8, "K": 9, "M": 10, "L": 11,
                "N": 12, "Q": 13, "P": 14, "S": 15, "R": 16,
                "T": 17, "W": 18,
                "V": 19, "Y": 20 ,"X" : 21}

dict_prot_len = len(dict_prot)

max_prot_len=1215

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
        if i%1000==0:
            print(i/dataframe.shape[0]*100,'%')
    XT = np.array(XT)
    print('Kinases encoded and ready for input')
    
    return(XT)



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



def gcnn(params,lr_value, windows_seq):

    atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features),dtype = 'float32')
    bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features),dtype = 'float32')
    edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
    XTinput = keras.layers.Input(shape = (1215, dict_prot_len))

    g1 = NeuralGraphHidden(params['conv_width'] , activ = None, bias = True , init = 'glorot_uniform')([atoms0,bonds,edges])
    g1 = BatchNormalization(momentum=0.6)(g1)
    g1 = Activation('relu')(g1)
    g1 = keras.layers.Dropout(0.25)(g1, training=True)
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
    encode_protein = keras.layers.Dropout(0.25)(encode_protein, training=True)
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
    fc2 = keras.layers.Dropout(0.25)(fc1, training=True) #this enables dropout also in test-time 
    fc2 = keras.layers.Dense(params['dense_size'],activation = None,kernel_regularizer=regularizers.l2(params['l2reg']))(fc2)
    fc2 = BatchNormalization(momentum=0.6)(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = keras.layers.Dropout(0.25)(fc2, training=True) #this enables dropout also in test-time 
    
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



graph_model=gcnn(p,0.001,32)
graph_model.load_weights('gcnn_model_test_dropoutweights.h5')


# ## Load validation data

# ### cold



df=pd.read_csv('./../main/data/clean/cold_clean.csv',index_col=0)



smiles = df['canonical_smiles']
X_atoms_val_cold, X_bonds_val_cold, X_edges_val_cold = tensorise_smiles(smiles, max_degree=5, max_atoms = 66)
num_molecules = X_atoms_val_cold.shape[0]
max_atoms = X_atoms_val_cold.shape[1]
max_degree = X_bonds_val_cold.shape[2]
num_atom_features = X_atoms_val_cold.shape[-1]
num_bond_features = X_bonds_val_cold.shape[-1]



seqs_val_cold = make_datasets(df)



y_val_cold=df['pkd']


# ### easy



df=pd.read_csv('./../main/data/clean/random_clean.csv',index_col=0)



smiles = df['canonical_smiles']
Xatoms_val_easy, Xbonds_val_easy, Xedges_val_easy = tensorise_smiles(smiles, max_degree=5, max_atoms = 66)
num_molecules = Xatoms_val_easy.shape[0]
max_atoms = Xatoms_val_easy.shape[1]
max_degree = Xbonds_val_easy.shape[2]
num_atom_features = Xatoms_val_easy.shape[-1]
num_bond_features = Xbonds_val_easy.shape[-1]



seqs_val_easy = make_datasets(df)



Y_val_easy=df['pkd']




# ## Make predictions

# ### easy

#easy
val_easy_preds=[]
times_of_prediction=50
pred_times=[]
for i in range(times_of_prediction):
    start_time=time.time()
    val_easy_preds.append(graph_model.predict([Xatoms_val_easy,Xbonds_val_easy,Xedges_val_easy,seqs_val_easy]))
    pred_times.append(time.time()-start_time)
    print('running! time left %.1f min'%(np.mean(pred_times)*(times_of_prediction-i)/60))
print('predictions made, converting to array now...')
val_easy_preds = np.array(val_easy_preds)



uncertainties_easy=np.std(val_easy_preds, axis=0).squeeze()
preds_easy=np.mean(val_easy_preds, axis=0).squeeze()
errors_val_easy=abs(np.subtract(preds_easy,Y_val_easy))



df_to_save=pd.DataFrame({'uncertainties':uncertainties_easy,'predictions':preds_easy,'y_true':Y_val_easy})
df_to_save.to_csv('dropout_uncertainties_easy.csv')


# ### cold



#cold
val_cold_preds=[]
times_of_prediction=50
pred_times=[]
for i in range(times_of_prediction):
    start_time=time.time()
    val_cold_preds.append(graph_model.predict([X_atoms_val_cold,X_bonds_val_cold,X_edges_val_cold,seqs_val_cold]))
    pred_times.append(time.time()-start_time)
    print('running! time left %.1f min'%(np.mean(pred_times)*(times_of_prediction-i)/60))
print('predictions made, converting to array now...')
val_cold_preds = np.array(val_cold_preds)


# In[13]:


uncertainties_cold=np.std(val_cold_preds, axis=0).squeeze()
preds_cold=np.mean(val_cold_preds, axis=0).squeeze()
errors_val_cold=abs(np.subtract(preds,Y_val_cold))


# In[14]:


df_to_save=pd.DataFrame({'uncertainties':uncertainties_cold,'predictions':preds_cold,'y_true':Y_val_cold})
df_to_save.to_csv('dropout_uncertainties_cold.csv')


# ### Load predictions

# ### cold



df=pd.read_csv('dropout_uncertainties_cold.csv',index_col=0)
uncertainties=np.array(df.uncertainties)
uncertainties_norm=[(i/max(uncertainties)) for i in uncertainties]
preds=np.array(df.predictions)
Y_val_cold=np.array(df.y_true)
errors_val=abs(np.subtract(preds,Y_val_cold))



print('MSE: %s'%(np.mean(errors_val**2)))

print('pearson:',(scipy.stats.pearsonr(preds,Y_val_cold)))

print('spearman:',(scipy.stats.spearmanr(preds,Y_val_cold)))

print('r^2:',(sklearn.metrics.r2_score(preds,Y_val_cold)))




plt.hist(uncertainties_norm,bins=50)
plt.show()


fig=plt.scatter(errors_val,uncertainties_norm,s=4)
plt.xlabel('abs(y_pred-y_true)', fontsize=14)
plt.ylabel('dropout variation', fontsize=14)
plt.xscale('log')
plt.show()




print(scipy.stats.pearsonr(errors_val,uncertainties_norm),sklearn.metrics.r2_score(errors_val,uncertainties_norm))



points_to_plot=[]
uncertainty_thres=[i for i in np.arange(min(uncertainties_norm),max(uncertainties_norm),0.01)]
for thres in uncertainty_thres:
    remaining_uncertainties=[]
    remaining_indices=[]
    for i, unc in enumerate(uncertainties_norm):
        if unc<thres:
            remaining_uncertainties.append(unc)
            remaining_indices.append(i)
    remaining_errors=[errors_val[i] for i in remaining_indices]
    points_to_plot.append(np.mean(remaining_errors))
fig=plt.scatter(uncertainty_thres,points_to_plot,s=10)
plt.xlabel('uncertainty threshold', fontsize=14)
plt.ylabel('mean error', fontsize=14)
plt.show()


# ### easy




df=pd.read_csv('dropout_uncertainties_easy.csv',index_col=0)
uncertainties=np.array(df.uncertainties)
uncertainties_norm=[(i/max(uncertainties)) for i in uncertainties]
preds=np.array(df.predictions)
Y_val_cold=np.array(df.y_true)
errors_val=abs(np.subtract(preds,Y_val_cold))



print('MSE: %s'%(np.mean(errors_val**2)))

print('pearson:',(scipy.stats.pearsonr(preds,Y_val_cold)))

print('spearman:',(scipy.stats.spearmanr(preds,Y_val_cold)))

print('r^2:',(sklearn.metrics.r2_score(preds,Y_val_cold)))



plt.hist(uncertainties_norm,bins=50)
plt.show()



fig=plt.scatter(errors_val,uncertainties_norm,s=4)
plt.xlabel('abs(y_pred-y_true)', fontsize=14)
plt.ylabel('dropout variation', fontsize=14)
plt.xscale('log')
plt.show()



print(scipy.stats.pearsonr(errors_val,uncertainties_norm),sklearn.metrics.r2_score(errors_val,uncertainties_norm))



points_to_plot=[]
uncertainty_thres=[i for i in np.arange(0.1,0.8,0.02)]
for thres in uncertainty_thres:
    remaining_uncertainties=[]
    remaining_indices=[]
    for i, unc in enumerate(uncertainties_norm):
        if unc<thres:
            remaining_uncertainties.append(unc)
            remaining_indices.append(i)
    remaining_errors=[errors_val[i] for i in remaining_indices]
    points_to_plot.append(np.mean(remaining_errors))
fig=plt.scatter(uncertainty_thres,points_to_plot,s=10)
plt.xlabel('uncertainty threshold', fontsize=14)
plt.ylabel('mean absolute error', fontsize=14)
plt.show()





