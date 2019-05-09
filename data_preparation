# Kinases-drug Kd binding datasets report
### Christos Fotis (chfoti@gmail.com)
### Panagiotis Terzopoulos (p.d.terzopoulos@gmail.com)
### BSLab NTUAMarch 2019

## Introduction
This report concerns the gathering and organizing of the various protein kinases-micromolecules binding datasets that exist for free in the web and in the literature as for the date of this report's writing. The various data of experimental biological labs around the world that comes out publicly available is unfortunately not yet formatted universally. The differences are usually spotted either on the compound or protein representations (i.e. different type of SMILES formatting, custom IDs and numbering, lack of unique UniprotKB IDs etc) or on the way each lab measures the affinity metric. In this work, an effort has been made to clean and format universally the binding affinity data between protein kinases and ligands, that comes from different sources, experiments, labs and is expressed in various metrics such as Kd, Ki, IC50 etc.

The goal of this work is to create a unique overall dataset that will contain the most probable Kd values of all the interactions between kinases and drugs found in the literature, specifically formatted to be easy to use for machine learning applications and competitions such as the [IDG-DREAM challenge](https://www.synapse.org/#!Synapse:syn15667962/wiki/583305) without any further work.

## Useful data in the literature
### DTC dataset 
Drug Target Commons (DTC) is a publicly available web platform (databasewith user interface) for community-driven bioactivity data integration and stan-dardization for comprehensive mapping, reuse and analysis of compound–targetinteraction profiles.

DTC  contains  1509  different  types  of  bioactivity  data  between  drugs  andkinases gathered from the literature and other open access databases.  Out ofthose  types  4  are  considered  to  be  dissociation  constants  (Kd,  KD,  KDAPP,PKD). The drug-kinase pairs in DTC that have at least one of those four typesare 85475 (not unique).  After filtering for NaN values as well as for Kds andunits that were not concentrations (not in PKD) 2895 pairs were removed.  Outof the 82580 pairs remaining, 2308 were removed due to having no protein ID.These 2308 pairs consist of 617 unique proteins that we have to fix.  In addition2821 pairs were missing the inchikey identifier making the process of acquiringtheir  smiles  a  bit  harder.   1247  pairs  were  mapped  to  smiles  based  on  theirChemblID and chemical name while the rest 1574 were mapped to their smileusing the UNC dataset (see section 2.7) that we extracted from a pdf.  Fromthis approach all but 2 drugs (13 pairs) (were restricted drugs) out of the  550that had no inchi keys were mapped to their smiles.

Next, the pairs that contained inchikeys were mapped to their smiles usingChembl.  On this front, 186 drugs involved in 767 interactions were not found inChembl.  128/186 were mapped to their smiles using BindingDB. The remaining58 drugs involved in 586 interactions were removed.

In  DTC  there  exist  3144  pairs  of  data  points  (compounds)  with  multipletarget proteins separated by comma.  Those data points were kept by splittingeach data point into multiple interactions creating 11529 interactions.  Althoughthis approach might not be the cleanest we don’t think it will cause any issues,because in the final step we filter to keep only the unique pairs based on thestandard deviation of the Kd values for multiple pairs.

The amino acid sequences for the proteins in the pairs were all collected from UniprotKB using the uniprot IDs of the pairs as identifiers.

Next, after observing the range of Kd values, the max value for Kd was setto 10000 NM and the min to 0.001 NM resulting in a Pkd metric in the rangeof 5 to 12.

Finally,  because  DTC  reports  the  sign  for  each  interaction  as  well  (<, >,=)  some  inconsistencies  were  removed.   That  is  if  the  sign  was<and  thevalue between 100 (pkd 7) and 10000 (pkd 5) NM the pair was removed dueto uncertainty.  Pairs with Kd<100 and sign<were kept being considered as”bound”.  In addition, if the sign was>and the value was<1000(pkd >6) thepair was removed.  Pairs with sign>and value>1000(pkd <6) were left in asthey were considered inactive.

Note:  These ’sign’ filters are applied to every dataset where is needed.

### PKIS dataset
The  Published  Kinase  Inhibitor  Set  (PKIS)  is  a  collection  of  376  compoundsthat  have  been  made  available  by  GSK  for  screening  by  external  groups;  allcompounds have been published in the scientific literature.

The PKIS contains 3719 interactions, out of which only 1084 concern Kd values. We recover the drug SMILES by mapping the PKIS InChI-keys on the CHEMBL database. 

The sequences are taken from UniprotKB and the Kd values are converted to pKd. The max pKd value is also set to pKd = 12 and larger values are removed.

Note: These filters and mappings are applied to every dataset.

### KKB database
The Kinase Knowledgebase (KKB) is Eidogen-Sertanty's database of kinase structure-activity and chemical synthesis data. The KKB contains over 1.8M kinase structure-activity data points reported in peer-reviewed journals and patents but one has to pay for it to be accessed. Rajan Sharma et al. [4] have extracted and made available publicly 258K structure activity data points and 76K associated unique chemical structures across eight kinase targets. 

Out of these 258K datapoints, one can extract 6227 Kd values associated with 233 compounds.

### DAVIS dataset [5]
This dataset contains the interactions (Kd) of 72 kinase inhibitors with 442 kinases covering >80% of the human catalytic protein kinome.

Small note: the drug names included in DAVIS dataset must be matched to InChI keys by mapping the CTS (Chemical Translation Service).

### BindingDB database
BindingDB is a public, web-accessible database of measured binding affinities, focusing chiefly on the interactions of protein considered to be drug-targets with small, drug-like molecules. BindingDB contains 1,587,753 binding data, for 7,235 protein targets and 710,301 small molecules.

Out of this data the interactions containing only kd metric have been kept (70300 interactions) out of which 69429 remained after mapping the InChI keys and applying the 'sign' and other filters. 

### HMS LINCS database
The Library of Integrated Network-based Cellular Signatures (LINCS) program is developing a library of molecular signatures that describes how different types of cells respond to a variety of agents that disrupt normal cellular functions. 

Under this database the Harvard medical school in collaboration with the NIH have made publicly available multiple kinases-compounds binding experiments using the [KINOMEscan](https://www.discoverx.com/services/drug-discovery-development-services/kinase-profiling/kinomescan) technology.

These datasets may contain directly Kd values or concentration of compound values that need to be multiplied by the control percentage to become Kd (units also differ between the different datasets in the website so that is also to be kept in mind). The datasets also identify the proteins and the compounds by custom LINCS IDs that need to be converted to UniprotKB IDs and canonical SMILES using other datasets provided in the website. 

Out of the 182 different csv files that can be downloaded from the LINCS database website and contain experiments with kinases using KINOMEscan, one can extract 401 proteins, 162 drugs and 42610 unique interactions between the protein-drug pairs out of the 63700 total values, throwing away nan values, restricted SMILES and after applying all the filters. 

### UNC dataset
Another useful kinases-ligands dataset can be found in [2] from the university of North Carolina. 

Extracting information out of this dataset is of particular difficulty due to the .pdf format of the data. However, interactions of great quality (relatively rare and structurally different molecules) can be found in here.

After a lot of custom export functions 361 drugs, 339 proteins and 1793 unique interactions were added.

## 'Useless' (non Kd) data in the literature
### GOSTAR database
Huge database which belongs to Excelra Knowledge Solutions (formerly GVK Informatics) company (part of the 1 billion dollars GVK group). This database is probably not freely available, although was not able to request a licence through the webpage, since it seems there is a bug when the website is trying to read the user's IP address.
Can be found [here](https://gostardb.com)

### EMD Millipore database
Pairwise assays between purified recombinant human kinases and a collection of small molecule inhibitors were carried out by the EMD Millipore Corporation (now known as Millipore Sigma) using a filter binding radioactive ATP transferase assay.

However, the metrics used for this database are the Km and S score, which unfortunately cannot be directly related to Kd. More information about this work can be found in [3].

### WOMBAT database
Overview can be found [here](https://www.researchgate.net/publication/229618698_Chemical_Informatics_WOMBAT_and_WOMBAT-PK_Bioactivity_Databases_for_Lead_and_Drug_Discovery)
The dataset can be found [here](http://dud.docking.org/wombat/)

### Kinase Inhibitor Resource (KIR) database
In [1] Anastasiadis et al. used a custom affinity metric that was cheaper for them to evaluate. The metric is defined as the average of two replicates which is shown as percent remaining kinase activity in the presence of the compound relative to solvent control. These values, unfortunately, cannot be directly interpreted as Kd values and thus they are 'useless' for this work. 

## Merging the datasets
After combining the data from all the different datasets all the interactions that exist several times should be treated so that they are included just once.

When merged initially the datasets contained 260571 pairs and 16871 unique drugs in total.

From the duplicates we kept the ones that their Kd value differ less than 10%. As a Kd value we kept their mean.
From the ones that exist more than twice, we kept the ones that their std/mean is less than 10%. As a Kd value we kept their mean.

Finally, 103741 unique interactions are included in our dataset.

### Universal SMILES format
In order to have a universal SMILES format (SMILES canonicalization) all the SMILES were converted to molecular graphs and back to SMILES using the RDkit anaconda tool. 

## Tools Used
It is obvious that the size of the datasets in this work is big enough to prevent any manual editing. All of the work was carried out using scripts. Most of the datasets manipulation was done using R in RStudio platform. Python was also used supportively to extract some information in specific cases as is the UNC, the LINCS and the PKIS datasets.

## References
[1]  Theonie Anastassiadis, Sean W Deacon, Karthik Devarajan, Haiching Ma,and  Jeffrey  R  Peterson.   Comprehensive  assay  of  kinase  catalytic  activ-ity  reveals  features  of  kinase  inhibitor  selectivity.Nature Biotechnology,29:1039–1045, 2011.

[2]  David  H.  Drewry,  Carrow  I.  Wells,  David  M.  Andrews,  Richard  Angell,Hassan Al-Ali, Alison D. Axtman, Stephen J. Capuzzi, Jonathan M. Elkins,Peter Ettmayer, Mathias Frederiksen, Opher Gileadi, Nathanael Gray, AliceHooper, Stefan Knapp, Stefan Laufer, Ulrich Luecking, Michael Michaelides,Susanne M ̈uller, Eugene Muratov, R. Aldrin Denny, Kumar S. Saikatendu,Daniel K. Treiber, William J. Zuercher, and Timothy M. Willson.  Progresstowards a public chemogenomic set for protein kinases and a call for contri-butions.PLOS ONE, 12(8):1–20, 08 2017.

[3]  Yinghong GAO, Stephen P. DAVIES, Martin AUGUSTIN, Anna WOOD-WARD, Umesh A. PATEL, Robert KOVELMAN, and Kevin J. HARVEY.A broad activity screen in support of a chemogenomic map for kinase sig-nalling research and drug discovery.Biochem. J., 451:313–328, 2013.

[4]  Rajan Sharma, Stephan C. Sch ̈urer, and Steven M. Muskal.  High quality,small molecule-activity datasets for kinase research.F1000Research, 2018.

[5] Davis, M., Hunt, J., Herrgard, S., Ciceri, P., Wodicka, L., Pallares, G., Hocker, M., Treiber, D. and Zarrinkar, P. (2011). Comprehensive analysis of kinase inhibitor selectivity. Nature Biotechnology, 29(11), pp.1046-1051.
