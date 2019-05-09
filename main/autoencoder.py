import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
#%matplotlib inline
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Concatenate
from keras import regularizers
from keras.callbacks import History, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam

tf.set_random_seed(1); np.random.seed(1); random.seed(1)

input_dir = os.path.join(os.getcwd(),'data')

### read the input smiles data for the autoencoder
sm = pd.read_csv(os.path.join(input_dir,'smiles_test50_times.csv'),index_col=0)
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

X_train, Y_train = vectorize2(sm.x.values)

print(sm.iloc[1])

### Autoencoder model

input_shape = X_train.shape[1:]
output_dim = Y_train.shape[-1]
latent_dim = 128
lstm_dim = 128

unroll = False
encoder_inputs = Input(shape=input_shape)
encoder = LSTM(lstm_dim, return_state=True,
                unroll=unroll)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
states = Concatenate(axis=-1)([state_h, state_c])
neck = Dense(latent_dim, activation="relu")
neck_outputs = neck(states)

decode_h = Dense(lstm_dim, activation="relu")
decode_c = Dense(lstm_dim, activation="relu")
state_h_decoded =  decode_h(neck_outputs)
state_c_decoded =  decode_c(neck_outputs)
encoder_states = [state_h_decoded, state_c_decoded]
decoder_inputs = Input(shape=input_shape)
decoder_lstm = LSTM(lstm_dim,
                    return_sequences=True,
                    unroll=unroll
                   )
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(output_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
#Define the model, that inputs the training vector for two places, and predicts one character ahead of the input
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())

### Training

h = History()
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=10, min_lr=0.000001, verbose=1, min_delta=1e-5)

opt=Adam(lr=0.005) #Default 0.001
model.compile(optimizer=opt, loss='categorical_crossentropy')

model.fit([X_train,X_train],Y_train,
                    epochs=1000,
                    batch_size=256,
                    callbacks=[h, rlr],
                    shuffle=True)


### define check function for the accuracy of the decoder

def check(samples,timesteps,ground,predictions):
    accs=[]
    for i in range(0,samples):
        k=0
        l=0
        for j in range(0,timesteps):
            if np.count_nonzero(ground[i,j]) != 0:
                l=l+1
                if np.argmax(predictions[i,j]) == np.argmax(ground[i,j]):
                    k=k+1
        acc = k/l
        #print(acc)
        accs.append(acc)
    return(accs)

### check the accuracy of the autoencoder in test_smiles.csv

sm_test = pd.read_csv(os.path.join(input_dir,'test_smiles.csv'),index_col=0)
X_test, Y_test = vectorize2(sm_test.x.values)
Ypred = model.predict([X_test,X_test])
test_rec = check(25,101,Y_test,Ypred)
np.mean(test_rec)
### Decouple the encoder
smiles_to_latent_model = Model(encoder_inputs, neck_outputs)
#test_encodings = smiles_to_latent_model.predict(X_test)
#np.save(os.path.join(input_dir,'test_encodings_org.npy'),test_encodings)

### Save the models
model.save(os.path.join(input_dir,'test_autoencoder.h5'))
smiles_to_latent_model.save(os.path.join(input_dir,'test_encoder.h5'))
