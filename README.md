# strainr: Strain-resolved classification and assembly primer 

### Intro

Strainr is a k-mer based metagenomic classifier that uses k-mer distributions and a naive Bayesian classifier to assign strain labels to DNA sequences. Strainr  
is fast - database creation and classification of a metagenomic sample takes only minutes and can be run on a laptop.





### 1. Building a database

The default method for database creation involves choosing a single species - this is most likely done after initial investigation of species-level classification.

```
python strainr_database.py --taxid [species-taxid] --out [database-name]
```


### 2. Classification

```
python strainr_classify.py --db [database-name] --out [output-directory] [fastq-file]
```

