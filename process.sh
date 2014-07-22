#!/bin/bash

mkdir -p $1
NUM=`ls -n result/$1/log-* | wc -l`
echo $NUM
for n in `seq 0 $((NUM-1))`
do
  awk '/test err/ { print $3; }' result/$1/log-$n > result/$1/test-$n
  awk '/train err/ { print $3; }' result/$1/log-$n > result/$1/train-$n
  # awk '/test time/ { print $5; }' $1/record-$n > $1/testtime-$n
done
LAST=$NUM
if [ ! -z "$2" ] 
  then
    LAST=$2
fi
python plot.py $1 $NUM $LAST
open plot.pdf