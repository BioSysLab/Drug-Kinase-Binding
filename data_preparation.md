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
