#!/bin/bash
read -p "Enter source, budget, method=none/selfSim/testSim, strategy=fully/partial, window, continuous flag, beta, tune=norm/none: " s n f l w c b t
echo
python active_labeling.py -s $s -n $n -f $f -l $l -w $w -c $c -b $b -t $t;
