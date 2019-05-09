#!/usr/bin/env python
# coding: utf-8

# ###  Similarity test using RDKit and different drug fingerprints 

# this is very useful: http://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
# 
# this may be useful: https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf

# In[9]:


import urllib
import os
import numpy as np
import pandas as pd
import sys
from sys import getsizeof
import scipy
from scipy import stats
    
import csv

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw

from rdkit.Chem.Fingerprints import FingerprintMols


# In[10]:


# # This runs through the molecules in a SMILES file and returns
# # a list of Molecules.
# def process_smiles_file(filename):
#     molecules = []
#     with open(filename, "r") as infile:
#         for smi in infile:
#             m = Chem.MolFromSmiles(smi)
#             if m is None:
#                 continue
#             molecules.append(m)
#     return molecules


# In[11]:


# molecules=process_smiles_file('all_smiles_with_test_v2.txt')


# In[12]:


with open("input_file.txt", "r") as f:
    lines=f.readlines()


# In[13]:


molecules=[]
for i,smi in enumerate(lines):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        print('smile with number:',i,'could not be converted to molecule')
        continue
    molecules.append(m)


# In[14]:


len(molecules)


# ### topological fingerprints

# In[15]:


#topological fingerprints
fps_topol = [FingerprintMols.FingerprintMol(x) for x in molecules]
similarities_topol=np.zeros(shape=((len(fps_topol),len(fps_topol))))


# In[16]:


#compute similarities.  Comment this section if only the fingerprints are needed
for i in range(len(fps_topol)):
    for j in range(len(fps_topol)):
        if i>j:
            similarities_topol[i][j]=DataStructs.FingerprintSimilarity(fps_topol[i],fps_topol[j]) #default is the tanimoto similarity
            similarities_topol[j][i]=similarities_topol[i][j]
        elif i==j:
            similarities_topol[i][j]=1
        #for other similarity metrics use for example DataStructs.FingerprintSimilarity(fps[0],fps[1], metric=DataStructs.DiceSimilarity)
    if i%500==0:
        print('running:',i/len(fps_topol)*100,'%')


# In[ ]:


df = pd.DataFrame(similarities_topol)
print(df)


# In[ ]:


df = pd.DataFrame(similarities_topol)
df.to_csv('similarities_topol.csv')


# ### Maccs keys fingerprints

# In[ ]:


from rdkit.Chem import MACCSkeys
fps_maccs = [MACCSkeys.GenMACCSKeys(x) for x in molecules]
similarities_maccs=np.zeros(shape=((len(fps_maccs),len(fps_maccs))))


# In[ ]:


#compute similarities.  Comment this section if only the fingerprints are needed
for i in range(len(fps_maccs)):
    for j in range(len(fps_maccs)):
        if i>j:
            similarities_maccs[i][j]=DataStructs.FingerprintSimilarity(fps_maccs[i],fps_maccs[j]) #default is the tanimoto similarity
            similarities_maccs[j][i]=similarities_maccs[i][j]
        elif i==j:
            similarities_maccs[i][j]=1
        #for other similarity metrics use for example DataStructs.FingerprintSimilarity(fps[0],fps[1], metric=DataStructs.DiceSimilarity)
    if i%500==0:
        print('running:',i/len(fps_maccs)*100,'%')


# In[ ]:


df = pd.DataFrame(similarities_maccs)
df.to_csv('similarities_maccs.csv')


# ### Atom pairs fingerprints

# In[ ]:


from rdkit.Chem.AtomPairs import Pairs
fps_pairs = [Pairs.GetAtomPairFingerprint(x) for x in molecules]
similarities_pairs=np.zeros(shape=((len(fps_pairs),len(fps_pairs))))


# In[ ]:


#compute similarities.  Comment this section if only the fingerprints are needed
for i in range(len(fps_pairs)):
    for j in range(len(fps_pairs)):
        if i>j:
            similarities_pairs[i][j]=DataStructs.DiceSimilarity(fps_pairs[i],fps_pairs[j]) #default is the Dice similarity for these fps
            similarities_pairs[j][i]=similarities_pairs[i][j]
        elif i==j:
            similarities_pairs[i][j]=1
    if i%500==0:
        print('running:',i/len(fps_pairs)*100,'%')


# In[ ]:


df = pd.DataFrame(similarities_pairs)
df.to_csv('similarities_pairs.csv')


# ### Topological torsion descriptors

# In[ ]:


from rdkit.Chem.AtomPairs import Torsions
fps_tts = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(x) for x in molecules]
similarities_tts=np.zeros(shape=((len(fps_tts),len(fps_tts))))


# In[ ]:


#compute similarities.  Comment this section if only the fingerprints are needed
for i in range(len(fps_tts)):
    for j in range(len(fps_tts)):
        if i>j:
            similarities_tts[i][j]=DataStructs.DiceSimilarity(fps_tts[i],fps_tts[j]) #default is the Dice similarity for these fps
            similarities_tts[j][i]=similarities_tts[i][j]
        elif i==j:
            similarities_tts[i][j]=1
    if i%500==0:
        print('running:',i/len(fps_tts)*100,'%')


# In[ ]:


df = pd.DataFrame(similarities_tts)
df.to_csv('similarities_tts.csv')


# ### Morgan Fingerprints
# (Circular Fingerprints: these are the same as ECFPx where x is the diameter (half radius) in morgan fps)

# In[ ]:


from rdkit.Chem import AllChem

fps_ecfp4 = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in molecules] #could also be not a bit vector and use Dice similarity
similarities_ecfp4=np.zeros(shape=((len(fps_ecfp4),len(fps_ecfp4))))

fps_ecfp6 = [AllChem.GetMorganFingerprintAsBitVect(x,3,nBits=1024) for x in molecules]
similarities_ecfp6=np.zeros(shape=((len(fps_ecfp6),len(fps_ecfp6))))

fps_fcfp4 = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024,useFeatures=True) for x in molecules]
similarities_fcfp4=np.zeros(shape=((len(fps_fcfp4),len(fps_fcfp4))))

fps_fcfp6 = [AllChem.GetMorganFingerprintAsBitVect(x,3,nBits=1024,useFeatures=True) for x in molecules]
similarities_fcfp6=np.zeros(shape=((len(fps_fcfp6),len(fps_fcfp6))))


# In[ ]:


#compute similarities.  Comment this section if only the fingerprints are needed
for i in range(len(fps_ecfp4)):
    for j in range(len(fps_ecfp4)):
        if i>j:
            similarities_ecfp4[i][j]=DataStructs.FingerprintSimilarity(fps_ecfp4[i],fps_ecfp4[j]) 
            similarities_ecfp6[i][j]=DataStructs.FingerprintSimilarity(fps_ecfp6[i],fps_ecfp6[j])
            similarities_fcfp4[i][j]=DataStructs.FingerprintSimilarity(fps_fcfp4[i],fps_fcfp4[j]) 
            similarities_fcfp6[i][j]=DataStructs.FingerprintSimilarity(fps_fcfp6[i],fps_fcfp6[j])
            similarities_ecfp4[j][i]=similarities_ecfp4[i][j]
            similarities_ecfp6[j][i]=similarities_ecfp6[i][j]
            similarities_fcfp4[j][i]=similarities_fcfp4[i][j]
            similarities_fcfp6[j][i]=similarities_fcfp6[i][j]
        elif i==j:
            similarities_ecfp4[i][j]=1
            similarities_ecfp6[i][j]=1
            similarities_fcfp4[i][j]=1
            similarities_fcfp6[i][j]=1
    if i%500==0:
        print('running:',i/len(fps_fcfp4)*100,'%')


# In[ ]:


df = pd.DataFrame(similarities_ecfp4)
df.to_csv('similarities_ecfp4.csv')


# In[ ]:


df = pd.DataFrame(similarities_ecfp6)
df.to_csv('similarities_ecfp6.csv')


# In[ ]:


df = pd.DataFrame(similarities_fcfp4)
df.to_csv('similarities_fcfp4.csv')


# In[ ]:


df = pd.DataFrame(similarities_fcfp6)
df.to_csv('similarities_fcfp6.csv')


# ### Fragle: 
# that's a similarity metric (like Tanimoto or Dice)
# presented here:
# https://github.com/rdkit/UGM_2013/blob/master/Presentations/Hussain.Fraggle.pdf
# 
# the post below claims that fragle is the most complementary with the other metrics (finds similarities where the others don't)
# https://rdkit.blogspot.com/2013/11/comparing-fraggle-to-other-fingerprints.html
# 

# In[ ]:


#for now fraggle doesn't work with our smiles


# In[ ]:


# from rdkit.Chem import Fraggle
# from rdkit.Chem.Fraggle import FraggleSim

# similarities_fraggle=np.zeros(shape=((len(molecules),len(molecules))))
# for i in range(len(molecules)):
#     for j in range(len(molecules)):
#         if i>j:
#             try:
#                 similarities_fraggle[i][j]=Chem.Fraggle.FraggleSim.GetFraggleSimilarity(molecules[i],molecules[j]) #default is the Dice similarity for these fps
#             except Exception as e:
#                 print(i,j)
#                 #print(e)
#                 pass
#         elif i==j:
#             similarities_fraggle[i][j]=1
#     if i%300==0:
#         print('running:',i/len(molecules)*100,'%')


# In[ ]:


# np.savetxt("similarities_fraggle.csv", similarities_fraggle, delimiter=",") #takes the same memory either we save it as txt or csv


# ## Check similarities and statistics
# (the test set drugs are the last 25)

# In[17]:


fps_types=['ecfp4','ecfp6','fcfp4','fcfp6','maccs','pairs','topol','tts']


# write a file with the basic numbers

# In[ ]:


# f=open('test_drugs_similarities.txt','w')
# f.write('')
# f.close()


# ### check how many above threshold for each fp

# In[ ]:


simil_thres=0.6

for i in range(len(similarities_ecfp4)-25,len(similarities_ecfp4)):
    max_val_fcfp4=-1
    max_val_fcfp6=-1
    max_val_ecfp4=-1
    max_val_ecfp6=-1
    max_val_pairs=-1
    max_val_maccs=-1
    max_val_topol=-1
    max_val_tts=-1

    count_ecfp4=0
    count_ecfp6=0
    count_fcfp4=0
    count_fcfp6=0
    count_pairs=0
    count_maccs=0
    count_topol=0
    count_tts=0
    for j in range(len(similarities_ecfp4)-25):
        if i>j:
            if similarities_fcfp4[i][j]>max_val_fcfp4 and similarities_fcfp4[i][j]!=1 :
                max_val_fcfp4=similarities_fcfp4[i][j]
                max_i_fcfp4=i
                max_j_fcfp4=j
            if similarities_fcfp6[i][j]>max_val_fcfp6 and similarities_fcfp6[i][j]!=1 :
                max_val_fcfp6=similarities_fcfp6[i][j]
                max_i_fcfp6=i
                max_j_fcfp6=j
            if similarities_ecfp4[i][j]>max_val_ecfp4 and similarities_ecfp4[i][j]!=1 :
                max_val_ecfp4=similarities_ecfp4[i][j]
                max_i_ecfp4=i
                max_j_ecfp4=j
            if similarities_ecfp6[i][j]>max_val_ecfp6 and similarities_ecfp6[i][j]!=1 :
                max_val_ecfp6=similarities_ecfp6[i][j]
                max_i_ecfp6=i
                max_j_ecfp6=j
            if similarities_pairs[i][j]>max_val_pairs and similarities_pairs[i][j]!=1 :
                max_val_pairs=similarities_pairs[i][j]
                max_i_pairs=i
                max_j_pairs=j
            if similarities_maccs[i][j]>max_val_maccs and similarities_maccs[i][j]!=1 :
                max_val_maccs=similarities_maccs[i][j]
                max_i_maccs=i
                max_j_maccs=j
            if similarities_topol[i][j]>max_val_topol and similarities_topol[i][j]!=1 :
                max_val_topol=similarities_topol[i][j]
                max_i_topol=i
                max_j_topol=j
            if similarities_tts[i][j]>max_val_tts and similarities_tts[i][j]!=1 :
                max_val_tts=similarities_tts[i][j]
                max_i_tts=i
                max_j_tts=j

            if similarities_ecfp4[i][j]>simil_thres and similarities_ecfp4[i][j]<1:
                count_ecfp4+=1
            if similarities_ecfp6[i][j]>simil_thres and similarities_ecfp6[i][j]<1:
                count_ecfp6+=1
            if similarities_fcfp6[i][j]>simil_thres and similarities_fcfp6[i][j]<1:
                count_fcfp6+=1
            if similarities_fcfp4[i][j]>simil_thres and similarities_fcfp4[i][j]<1:
                count_fcfp4+=1
            if similarities_maccs[i][j]>simil_thres and similarities_maccs[i][j]<1:
                count_maccs+=1
            if similarities_pairs[i][j]>simil_thres and similarities_pairs[i][j]<1:
                count_pairs+=1
            if similarities_topol[i][j]>simil_thres and similarities_topol[i][j]<1:
                count_topol+=1
            if similarities_tts[i][j]>simil_thres and similarities_tts[i][j]<1:
                count_tts+=1

    count_ecfp4=int(round(count_ecfp4/2))
    count_ecfp6=int(round(count_ecfp6/2))
    count_fcfp6=int(round(count_fcfp6/2))
    count_fcfp4=int(round(count_fcfp4/2))
    count_maccs=int(round(count_maccs/2))
    count_pairs=int(round(count_pairs/2))
    count_topol=int(round(count_topol/2))
    count_tts=int(round(count_tts/2))

#     print('drug:',i,'\n')
#     print('max_ecfp4:','%.2f'%max_val_ecfp4,'  with the drug:  ',max_j_ecfp4)
#     print('ecfp4 similarities above',simil_thres,': ',count_ecfp4)
#     print('\n')
#     print('max_ecfp6:','%.2f'%max_val_ecfp6,'  with the drug:  ',max_j_ecfp6)
#     print('ecfp6 similarities above',simil_thres,': ',count_ecfp6)
#     print('\n')
#     print('max_fcfp6:','%.2f'%max_val_fcfp6,'  with the drug:  ',max_j_fcfp6)
#     print('fcfp6 similarities above',simil_thres,': ',count_fcfp6)
#     print('\n')
#     print('max_fcfp4:','%.2f'%max_val_fcfp4,'  with the drug:  ',max_j_fcfp4)
#     print('fcfp4 similarities above',simil_thres,': ',count_fcfp4)
#     print('\n')
#     print('max_maccs:','%.2f'%max_val_maccs,'  with the drug:  ',max_j_maccs)
#     print('maccs similarities above',simil_thres,': ',count_maccs)
#     print('\n')
#     print('max_pairs:','%.2f'%max_val_pairs,'  with the drug:  ',max_j_pairs)
#     print('pairs similarities above',simil_thres,': ',count_pairs)
#     print('\n')
#     print('max_topol:','%.2f'%max_val_topol,'  with the drug:  ',max_j_topol)
#     print('topol similarities above',simil_thres,': ',count_topol)
#     print('\n')
#     print('max_tts:','%.2f'%max_val_tts,'  with the drug:  ',max_j_tts)
#     print('tts similarities above',simil_thres,': ',count_tts)
#     print('\n')

#     f=open('test_drugs_similarities.txt','a')
#     f.write('COMPOUND OF TEST-SET: %s\n'%i)
#     f.write('max_ecfp4: %.2f   with the drug:  %s\n'%(max_val_ecfp4,max_j_ecfp4))
#     f.write('ecfp4 similarities above %s: %s \n\n'%(simil_thres,count_ecfp4))
#     f.write('max_ecfp6: %.2f   with the drug:  %s\n'%(max_val_ecfp6,max_j_ecfp6))
#     f.write('ecfp6 similarities above %s: %s \n\n'%(simil_thres,count_ecfp6))
#     f.write('max_fcfp4: %.2f   with the drug:  %s\n'%(max_val_fcfp4,max_j_fcfp4))
#     f.write('fcfp4 similarities above %s: %s \n\n'%(simil_thres,count_fcfp4))
#     f.write('max_fcfp6: %.2f   with the drug:  %s\n'%(max_val_fcfp6,max_j_fcfp6))
#     f.write('fcfp6 similarities above %s: %s \n\n'%(simil_thres,count_fcfp6))
#     f.write('max_maccs: %.2f   with the drug:  %s\n'%(max_val_maccs,max_j_maccs))
#     f.write('maccs similarities above %s: %s \n\n'%(simil_thres,count_maccs))
#     f.write('max_topol: %.2f   with the drug:  %s\n'%(max_val_topol,max_j_topol))
#     f.write('topol similarities above %s: %s \n\n'%(simil_thres,count_topol))
#     f.write('max_pairs: %.2f   with the drug:  %s\n'%(max_val_pairs,max_j_pairs))
#     f.write('pairs similarities above %s: %s \n\n'%(simil_thres,count_pairs))
#     f.write('max_tts: %.2f   with the drug:  %s\n'%(max_val_tts,max_j_tts))
#     f.write('tts similarities above %s: %s \n\n\n'%(simil_thres,count_tts))
#     f.close()


# #### check how many drugs of the train set are similar to every test drug for every fp

# In[ ]:


simil_thres=0.7
similar_train_drugs=[]

for i in range(len(similarities_ecfp4)-25,len(similarities_ecfp4)):
    max_val_fcfp4=-1
    max_val_fcfp6=-1
    max_val_ecfp4=-1
    max_val_ecfp6=-1
    max_val_pairs=-1
    max_val_maccs=-1
    max_val_topol=-1
    max_val_tts=-1

    count_ecfp4=0
    count_ecfp6=0
    count_fcfp4=0
    count_fcfp6=0
    count_pairs=0
    count_maccs=0
    count_topol=0
    count_tts=0
    for j in range(len(similarities_ecfp4)-25):
        if i>j:
            if similarities_ecfp4[i][j]>simil_thres and similarities_ecfp4[i][j]<1:
                count_ecfp4+=1
            if similarities_ecfp6[i][j]>simil_thres and similarities_ecfp6[i][j]<1:
                count_ecfp6+=1
            if similarities_fcfp6[i][j]>simil_thres and similarities_fcfp6[i][j]<1:
                count_fcfp6+=1
            if similarities_fcfp4[i][j]>simil_thres and similarities_fcfp4[i][j]<1:
                count_fcfp4+=1
            if similarities_maccs[i][j]>simil_thres and similarities_maccs[i][j]<1:
                count_maccs+=1
            if similarities_pairs[i][j]>simil_thres and similarities_pairs[i][j]<1:
                count_pairs+=1
            if similarities_topol[i][j]>simil_thres and similarities_topol[i][j]<1:
                count_topol+=1
            if similarities_tts[i][j]>simil_thres and similarities_tts[i][j]<1:
                count_tts+=1

    similar_train_drugs.append(max(count_ecfp4,count_ecfp6,count_fcfp4,count_fcfp6))
    #every position in the list corresponds to a test-set drug and the value of it corresponds to the number of training drugs
    #it is similar to for the chosen threshold


# In[ ]:


print(similar_train_drugs)


# In[ ]:


more_drugs_than_this_list_index=[]
for j in range(0,max(similar_train_drugs)+1):
    count=0
    for i in similar_train_drugs:
        if i==j:
            count+=1
    more_drugs_than_this_list_index.append(count)
    #if x the the value of each element in this list and y the position (index) of every element in the list respectively then
    #x test drugs have more than 'simil_thres' similarity with y train drugs


# In[ ]:


print(more_drugs_than_this_list_index)


# In[ ]:


print('x test drugs have more than %s similarity with y train drugs'%simil_thres)
for i,elem in enumerate(more_drugs_than_this_list_index):
    print(elem,i)


# ### compute correlation between fps
# (pubchem not included)

# In[18]:


filenames=[]
for fps_type in fps_types:
    filenames.append('F:/panost/main_work/fingerprints-similarity/similarities_'+fps_type+'.csv')


# In[20]:


print('loading files...')
d={}
for fps_type,file in zip(fps_types,filenames):
    f=pd.read_csv(file)
    f=f.drop('Unnamed: 0',axis=1)
    d["similarities_{0}".format(fps_type)]=f
    del f
    print(file,'loaded')


# In[22]:


correl_ecfp6_ecfp4=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_ecfp4'].values.flatten()))

correl_ecfp6_fcfp6=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_fcfp6'].values.flatten()))

correl_ecfp6_fcfp4=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_fcfp4'].values.flatten()))

correl_ecfp6_maccs=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_maccs'].values.flatten()))

correl_ecfp6_pairs=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_pairs'].values.flatten()))

correl_ecfp6_topol=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_ecfp6_tts=list(scipy.stats.spearmanr(d['similarities_ecfp6'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_ecfp4_fcfp6=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_fcfp6'].values.flatten()))

correl_ecfp4_fcfp4=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_fcfp4'].values.flatten()))

correl_ecfp4_maccs=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_maccs'].values.flatten()))

correl_ecfp4_pairs=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_pairs'].values.flatten()))

correl_ecfp4_topol=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_ecfp4_tts=list(scipy.stats.spearmanr(d['similarities_ecfp4'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_fcfp6_maccs=list(scipy.stats.spearmanr(d['similarities_fcfp6'].values.flatten(),d['similarities_maccs'].values.flatten()))

correl_fcfp6_pairs=list(scipy.stats.spearmanr(d['similarities_fcfp6'].values.flatten(),d['similarities_pairs'].values.flatten()))

correl_fcfp6_topol=list(scipy.stats.spearmanr(d['similarities_fcfp6'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_fcfp6_tts=list(scipy.stats.spearmanr(d['similarities_fcfp6'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_fcfp4_fcfp6=list(scipy.stats.spearmanr(d['similarities_fcfp4'].values.flatten(),d['similarities_fcfp6'].values.flatten()))

correl_fcfp4_maccs=list(scipy.stats.spearmanr(d['similarities_fcfp4'].values.flatten(),d['similarities_maccs'].values.flatten()))

correl_fcfp4_pairs=list(scipy.stats.spearmanr(d['similarities_fcfp4'].values.flatten(),d['similarities_pairs'].values.flatten()))

correl_fcfp4_topol=list(scipy.stats.spearmanr(d['similarities_fcfp4'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_fcfp4_tts=list(scipy.stats.spearmanr(d['similarities_fcfp4'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_maccs_pairs=list(scipy.stats.spearmanr(d['similarities_maccs'].values.flatten(),d['similarities_pairs'].values.flatten()))

correl_maccs_topol=list(scipy.stats.spearmanr(d['similarities_maccs'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_maccs_tts=list(scipy.stats.spearmanr(d['similarities_maccs'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_pairs_topol=list(scipy.stats.spearmanr(d['similarities_pairs'].values.flatten(),d['similarities_topol'].values.flatten()))

correl_pairs_tts=list(scipy.stats.spearmanr(d['similarities_pairs'].values.flatten(),d['similarities_tts'].values.flatten()))

correl_topol_tts=list(scipy.stats.spearmanr(d['similarities_topol'].values.flatten(),d['similarities_tts'].values.flatten()))


# In[26]:


d2 = {'fingerprint': ['ecfp6', 'ecfp4','fcfp6','fcfp4','maccs','pairs','topol','tts'],       'ecfp6': [[1, 0],correl_ecfp6_ecfp4,correl_ecfp6_fcfp6,correl_ecfp6_fcfp4,correl_ecfp6_maccs,correl_ecfp6_pairs,correl_ecfp6_topol,correl_ecfp6_tts],      'ecfp4': [correl_ecfp6_ecfp4,[1,0],correl_ecfp4_fcfp6,correl_ecfp4_fcfp4,correl_ecfp4_maccs,correl_ecfp4_pairs,correl_ecfp4_topol,correl_ecfp4_tts],      'fcfp6': [correl_ecfp6_fcfp6,correl_ecfp4_fcfp6,[1,0],correl_fcfp4_fcfp6,correl_fcfp6_maccs,correl_fcfp6_pairs,correl_fcfp6_topol,correl_fcfp6_tts],      'fcfp4': [correl_ecfp6_fcfp4,correl_ecfp4_fcfp4,correl_fcfp4_fcfp6,[1,0],correl_fcfp4_maccs,correl_fcfp4_pairs,correl_fcfp4_topol,correl_fcfp4_tts],      'maccs': [correl_ecfp6_maccs,correl_ecfp4_maccs,correl_fcfp6_maccs,correl_fcfp4_maccs,[1,0],correl_maccs_pairs,correl_maccs_topol,correl_maccs_tts],      'pairs': [correl_ecfp6_pairs,correl_ecfp4_pairs,correl_fcfp6_pairs,correl_fcfp4_pairs,correl_maccs_pairs,[1,0],correl_pairs_topol,correl_pairs_tts],      'topol': [correl_ecfp6_topol,correl_ecfp4_topol,correl_fcfp6_topol,correl_fcfp4_topol,correl_maccs_topol,correl_pairs_topol,[1,0],correl_topol_tts],      'tts': [correl_ecfp6_tts,correl_ecfp4_tts,correl_fcfp6_tts,correl_fcfp4_tts,correl_maccs_tts,correl_pairs_tts,correl_topol_tts,[1,0]]}
df = pd.DataFrame(data=d2)


# In[28]:


df.to_csv('fingerprints_correlation.csv')


# In[29]:


df_fps_corr_loaded=pd.read_csv('fingerprints_correlation.csv')
df_fps_corr_loaded=df_fps_corr_loaded.drop('Unnamed: 0',axis=1) 


# In[30]:


df_fps_corr_loaded






