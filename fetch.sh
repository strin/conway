#!/bin/bash

mkdir -p result/$1
NUM=`ssh tianlins@jacob.stanford.edu -C "ls -n ~/scr/conway/state/execs/$1/*.exec/log | wc -l"`
for n in `seq 0 $((NUM-1))`
do
  scp tianlins@jacob.stanford.edu:~/scr/conway/state/execs/$1/$n.exec/log result/$1/log-$n
done
