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
from scipy import spatial

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
    g1 = keras.layers.Dropout(0.25)(g1)
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
    encode_protein = keras.layers.Dropout(0.25)(encode_protein)
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



graph_model=gcnn(p,0.001,32)
graph_model.load_weights('gcnn_model_weights_nodropout.h5')



decoupled_model = keras.Model(graph_model.inputs, graph_model.layers[-3].output)
decoupled_model.summary()


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

# In[67]:


preds_latent=decoupled_model.predict([Xatoms_val_easy,Xbonds_val_easy,Xedges_val_easy,seqs_val_easy])
print(preds_latent[0].shape)


# In[68]:


preds=graph_model.predict([Xatoms_val_easy,Xbonds_val_easy,Xedges_val_easy,seqs_val_easy])
preds=np.reshape(preds,6500)
print(preds.shape)


# In[69]:


errors=abs(np.subtract(preds,Y_val_easy))


# ### Load predictions



label_train=np.load('label_train.npy')
label_random=np.load('label_random.npy')
label_cold=np.load('label_cold.npy')

latent_train=np.load('latent_train.npy')
latent_random=np.load('latent_random.npy')
latent_cold=np.load('latent_cold.npy')
latent_all=np.vstack((latent_train,latent_cold))

predictions_train=np.load('predictions_train.npy')
predictions_train=np.reshape(predictions_train,predictions_train.shape[0])
predictions_random=np.load('predictions_random.npy')
predictions_random=np.reshape(predictions_random,predictions_random.shape[0])
predictions_cold=np.load('predictions_cold.npy')
predictions_cold=np.reshape(predictions_cold,predictions_cold.shape[0])




errors_random=abs(np.subtract(predictions_random,label_random))
errors_cold=abs(np.subtract(predictions_cold,label_cold))
errors_train=abs(np.subtract(predictions_train,label_train))



# ### Compute distances

# #### easy (random split)



val_distances=[]
loop_times=[]
for i,val_compound in enumerate(latent_random):
    similarities=[]
    start_time=time.time()
    for train_compound in latent_train:
        #similarities.append( 1 - spatial.distance.cosine(train_compound, val_compound))
        similarities.append(np.linalg.norm(train_compound - val_compound))
    loop_times.append(time.time()-start_time)
    if i%400==0:
        print('running! time left %.1f min'%(np.mean(loop_times)*(len(latent_random)-i)/60))
    val_distances.append(similarities)




np.save('random_val_distances_with_train.npy',val_distances)


# #### cold



val_distances=[]
loop_times=[]
for i,val_compound in enumerate(latent_cold):
    similarities=[]
    start_time=time.time()
    for train_compound in latent_train:
        #similarities.append( 1 - spatial.distance.cosine(train_compound, val_compound))
        similarities.append(np.linalg.norm(train_compound - val_compound))
    loop_times.append(time.time()-start_time)
    if i%400==0:
        print('running! time left %.1f min'%(np.mean(loop_times)*(len(latent_cold)-i)/60))
    val_distances.append(similarities)




np.save('cold_val_distances_with_train.npy',val_distances)


# ### Load distances



setting='random' #setting = random or cold



val_distances_loaded=np.load('%s_val_distances_with_train.npy'%setting,mmap_mode ='r')



val_distances_loaded=val_distances_loaded
val_distances_loaded.shape



all_means=[]
for dist in val_distances_loaded:
    all_means.append(np.mean(dist))
radius=np.mean(all_means)/6
print(radius)



rad=[3, 2, 1 ,0.67, 0.5, 0.3]




for radius in rad:

    uncertainties=[]
    loop_times=[]
    
    for i,distances in enumerate(val_distances_loaded):
        start_time=time.time()

#     # the number-inside-a-radius approach
#         radius_factor=1
#         close_points=np.count_nonzero(distances < radius*radius_factor)
#         while close_points==0:
#             radius_factor+=0.2
#             close_points=np.count_nonzero(distances < radius*radius_factor)
#         uncertainties.append(1/close_points)
#         save_word='number_of_neighbors_radius_%s'%radius


#     # the median-of-distances approach
#         uncertainties.append(np.median(distances))
#         save_word='median_distances'


     # the median-inside-a-radius-approach
        close_points_distances=[]
        radius_factor=1
        while not close_points_distances: #increase radius for points that have no neighbors
            close_points_distances=[point for point in distances if point < radius*radius_factor]
            radius_factor+=0.2
        uncertainties.append(np.median(close_points_distances))
        save_word='median_neighbor_distances_radius_%s'%radius


#         # the error of neighbors approach
#         close_points_errors=[]
#         radius_factor=1
#         while not close_points_errors: #increase radius for points that have no neighbors
#             close_points_errors=[errors_train[i] for i in range(len(errors_train)) if distances[i] < radius*radius_factor]
#             radius_factor+=0.2
#         uncertainties.append(np.median(close_points_errors))
#         save_word='error_of_neighbors_radius_%s'%radius



        loop_times.append(time.time()-start_time)
        if i%400==0:
            print('running! time left %.1f min'%(np.mean(loop_times)*(len(val_distances_loaded)-i)/60))

    # fix the -1
    max_uncer=max(uncertainties)
    mean_uncer=np.mean(uncertainties)
    print('max =',max_uncer)
    print('mean =',mean_uncer)

    if setting=='cold':
        df_to_save=pd.DataFrame({'uncertainties':uncertainties,'predictions':predictions_cold,'y_true':label_cold})
        df_to_save.to_csv('uncertainties/distance/%s/distance_uncertainties_%s.csv'%(setting,save_word))
    elif setting=='random':
        df_to_save=pd.DataFrame({'uncertainties':uncertainties,'predictions':predictions_random,'y_true':label_random})
        df_to_save.to_csv('uncertainties/distance/%s/distance_uncertainties_%s.csv'%(setting,save_word))
        
