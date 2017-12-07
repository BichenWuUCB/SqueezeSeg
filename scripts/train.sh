#!/bin/bash

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="train"
export LOG_DIR="/tmp/bichen/logs/squeezeseg/"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-gpu                      gpu id"
  echo "-image_set                (train|val)"
  echo "-log_dir                  Where to save logs."
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
      echo "-log_dir                  Where to save logs."
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

now=$(date +"%F_%H-%M-%S")
logdir="$LOG_DIR/$now/"
echo "Command to run tensorboard: tensorboard --logdir=$logdir"
sed -i "1i $logdir" train_dir.txt

python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
  --data_path=./data/ \
  --image_set=$IMAGE_SET \
  --train_dir="$logdir/train" \
  --net=$NET \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=$GPUID
