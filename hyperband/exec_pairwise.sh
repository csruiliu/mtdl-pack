#!/bin/bash
echo 'start pairwising'
END=31
START=7
for i in $(seq $START $END); 
do 
    touch 'exp_'$i.txt 
    python3 clean_gpu_cache.py
    rm -rf __pycache__
    SNG_A=`python3 pairwise_engine.py --single $i`  
    for j in $(seq $i $END);
    do
        python3 clean_gpu_cache.py
        rm -rf __pycache__
        SNG_B=`python3 pairwise_engine.py --single $j`
        python3 clean_gpu_cache.py
        rm -rf __pycache__
        PACK_AB=`python3 pairwise_engine.py --packmode --pack $i,$j` 
        echo "scale=4; ($SNG_A+$SNG_B-$PACK_AB)/($SNG_A+$SNG_B)" | bc >> 'exp_'$i.txt 
    done
done
