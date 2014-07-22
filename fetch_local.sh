#!/bin/bash

mkdir -p result/$1
NUM=`ls -n state/$1/*.exec/log | wc -l`
for n in `seq 0 $((NUM-1))`
do
  scp state/$1/$n.exec/log result/$1/log-$n
done
