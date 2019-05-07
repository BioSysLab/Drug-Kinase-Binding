# Kinases-drug Kd binding datasets report
### Christos Fotis (chfoti@gmail.com)
### Panagiotis Terzopoulos (p.d.terzopoulos@gmail.com)
### BSLab NTUAMarch 2019

## Introduction
This report concerns the gathering and organizing of the various protein kinases-micromolecules binding datasets that exist for free in the web and in the literature as for the date of this report's writing. The various data of experimental biological labs around the world that comes out publicly available is unfortunately not yet formatted universally. The differences are usually spotted either on the compound or protein representations (i.e. different type of SMILES formatting, custom IDs and numbering, lack of unique UniprotKB IDs etc) or on the way each lab measures the affinity metric. In this work, an effort has been made to clean and format universally the binding affinity data between protein kinases and ligands, that comes from different sources, experiments, labs and is expressed in various metrics such as Kd, Ki, IC50 etc.

The goal of this work is to create a unique overall dataset that will contain the most probable Kd values of all the interactions between kinases and drugs found in the literature, specifically formatted to be easy to use for machine learning applications and competitions such as the [IDG-DREAM challenge](https://www.synapse.org/#!Synapse:syn15667962/wiki/583305) without any further work.
