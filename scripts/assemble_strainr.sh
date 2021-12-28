#!/bin/zsh

for fdir in strainr_c3*q/
do
    curdir=${fdir}afastas/

    if [ -d $curdir ]; then
        cd ${fdir}afastas/
        echo $(pwd)

        for i in GCA*R1.fastq; do
            r1=$(basename $i)
            base=${r1%%_R1.fastq}
            r2=${base}_R2.fastq

            echo $r1 $r2 $base
            echo $r1 $r2 $base

            # Call spades
            spades.py -1 $r1 -2 $r2 -o $base -t 48

            # Post assembly stuff
            cd $base
            # echo " Calling bwa "
            # bwa index scaffolds.fasta && bwa mem -t 24 scaffolds.fasta ../$r1 ../$r2 -o bwa.sam && samtools sort -@ 24 bwa.sam -o bwa.bam && bwa index bwa.bam
            cp scaffolds.fasta nomb2_$base.fa
            cd ..
        done
    # in afastas now
    mkdir nomb2_bins
    cp */nomb2*fa nomb2_bins
    checkm lineage_wf -f checkm.tsv -t 8 --pplacer_threads 8 -x fa . logs --ali --nt --tab_table &

    echo " Finished $curdir "
    cd ../../
    fi
done

