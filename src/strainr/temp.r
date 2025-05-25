if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# BiocManager::install(c("phyloseq"))

library("phyloseq")
df <- phyloseq::import_biom("/media/mladen/kai/sarc/mini_analysis/bracken/bracken_summary.hd5")
df
