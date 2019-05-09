# Deep learning for Receptor/ligand binding affinity prediction epxressed with Kd
### Christos Fotis<sup>1</sup>, Panagiotis Terzopoulos<sup>1</sup>, Konstantinos Ntagiantas<sup>1</sup> and Leonidas G.Alexopoulos<sup>1,2</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

## Introduction
The goal of this work is to build a deep end-to-end regression model similar to Ozturk et al. [1], that takes as input the SMILES representation of a compound and the amino acid sequence of a protein and outputs the KD value of the compound-protein pair.
On this front, we investigated several deep architectures in order to represent the SMILES-sequence pairs in a latent space that best captures the nature of the binding affinity prediction. Furthermore, a considerable amount of our effort focused on augmenting our training dataset as much as possible, with all the compound-protein pairs having available KD values in the literature.
## Methods
### Data and augmentation
As an initial dataset we started from the one provided by the [IDG-Dream Challenge for Kinase binding 2019](https://www.synapse.org/#!Synapse:syn15667962/wiki/583305) (DTC dataset) and we augmented it using various compound-kinase binding datasets that are publicly available in the web and in the literature. Overall, compound-kinase pairs with KD values from DTC, BindingDB, KKB, PKIS, HMS LINCS and Davis et al. were combined to create the final dataset for training and validation. The final dataset consisted of over 105K unique drug-protein interactions labeled with the Kd affinity metric. A detailed report of this work can be found in [here](https://github.com/bsl-ntua/Drug-Kinase-Binding/blob/master/data_preparation.md).
### Models
We experimented with different end-to-end architectures that utilize different methods for the latent representation of the SMILES and a deep CNN for the latent representation of the amino acid sequences.
1. The first architecture used a 3 layer deep graph convolutional network similar to [2], to extract application specific neural fingerprints from the compound structures. These fingerprints were then concatenated with the output of a 3 layer deep CNN that encodes the amino acid sequences of the proteins. The combined feature vector was fed through 2 fully connected layers for the final KD prediction. Batch normalization layers and relu activations were used throughout the network except for the final prediction layer. In order to reduce overfitting, dropout and L2 regularization was used between the fully connected layers.
2. Regarding the second architecture, a deep LSTM autoencoder was first trained on the SMILES sequences of all the compounds in the training, validation and test sets. The autoencoder used as input the one-hot representation of the SMILES and was tasked to predict the next letter in the sequence. The output of the trained encoder can serve as a compressed latent representation of the SMILES space. The idea behind training an autoencoder first, is that the encoder learns to represent all the available SMILES (training and test) and the final model should perform better than a fully end-to-end architecture that has never seen the structures of the test set. Thus, the final model used as input the output of the encoder along with the one-hot encoded amino acid sequences. The sequences were encoded again using a 3 layer deep CNN and concatenated with the output of the encoder to build the final feature vector. This feature vector was then passed through 2 fully connected layers for the final KD prediction. Batch normalization layers and relu activations were used throughout the network except the final prediction layer. In order to reduce overfitting, dropout and L2 regularization was used between the fully connected layers. 
## Training and Evaluation
The final augmented dataset consisted of 105431 unique pkd values between 12041 compounds and 1690 kinases, with more than 70000 pairs having a pkd value close to 5 (KD = 10000 μM). In order to reduce the bias of the trained model towards inactive compounds we decided to filter the interactions resulting in a final 3:1 ratio between inactive (pkd<7) and active pairs (pkd>=7). 
For model evaluation and parameter tuning a competition specific 5-fold cross validation scheme was employed. More specifically, we identified 5 sets of compounds, with similarity profiles with the training set, almost identical to the similarity profiles of the test set. During each step of the cross-validation all interactions that included the compounds of the validation set were used for model evaluation and the rest for model training.
The data augmentation pipeline was implemented using R while the models were built in python using keras with tensorflow as back end. Training was performed on a NVIDIA GPU GTX-1080Ti.
## Results and discussion
The best predictions for the test set came out of the second architecture we implemented which included the encoder. Having in mind how difficult it is to really generalize to new compound scaffolds never previously seen during training [3], an encoder that has been trained to represent the combined train and test-set distribution is expected to boost performance when its encoded feature vector is fed for further training.     

We also used this model to participate in [IDG-Dream Challenge for Kinase binding](https://www.synapse.org/#!Synapse:syn15667962/wiki/583305)

