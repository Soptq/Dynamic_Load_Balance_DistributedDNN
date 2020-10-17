#!/bin/bash
if [[ $# -ne 5 && $# -ne 6 ]]
then
  echo ""
  echo "========================="
  echo "Usage: ./run.sh [WORLD_SIZE] [BATCH_SIZE] [EPOCH_SIZE] [LEARNING_RATE] [GPUSET] [DE(OPTIONAL)]"
  echo "========================="
  echo ""
  exit 0
fi

if [ $# -ne 6 ]
then
  DE="false"
else
  DE=$6
fi

EPOCH_SIZE=$3
LEARNING_RATE=$4
GPUSET=$5
WORLD_SIZE=$1
BATCH_SIZE=$2

MODEL_LIST="resnet densenet googlenet regnet"
DATASET_LIST="cifar10 cifar100"
DBS_LIST="true false"

for dbs in $DBS_LIST
  do
  for dataset in $DATASET_LIST
    do
    for model in $MODEL_LIST
      do
        echo ""
        echo "========================="
        echo "Running:"
        echo "python dbs.py -d false -ws $WORLD_SIZE -lr $LEARNING_RATE -b $BATCH_SIZE -e $EPOCH_SIZE -ds $dataset -dbs $dbs -m $model -ocp true -gpu $GPUSET -de $DE"
        echo "========================="
        echo ""
        eval "python dbs.py -d false -ws $WORLD_SIZE -lr $LEARNING_RATE -b $BATCH_SIZE -e $EPOCH_SIZE -ds $dataset -dbs $dbs -m $model -ocp true -gpu $GPUSET -de $DE"
        if [ $? -ne 0 ]
        then
          echo ""
          echo "========================="
          echo "FAILED AT DATASET $dataset, MODEL $model"
          echo "========================="
          echo ""
          exit 1
        fi
      done
    done
  done