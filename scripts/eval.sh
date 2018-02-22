#!/bin/bash

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="val"
export LOG_DIR="./log/"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/eval.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-gpu                      gpu id"
  echo "-image_set                (train|val)"
  echo "-log_dir                  Where to load models and save logs."
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-gpu                      gpu id"
      echo "-image_set                (train|val)"
      echo "-log_dir                  Where to load models and save logs."
      exit 0
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -image_set)
      export IMAGE_SET="$2"
      shift
      shift
      ;;
    -log_dir)
      export LOG_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

logdir="$LOG_DIR/"
traindir="$logdir/train/"
valdir="$logdir/eval_$IMAGE_SET"

python ./src/eval.py \
  --dataset=KITTI \
  --data_path=./data/ \
  --image_set=$IMAGE_SET \
  --eval_dir="$valdir" \
  --checkpoint_path="$traindir" \
  --net=$NET \
  --gpu=$GPUID
