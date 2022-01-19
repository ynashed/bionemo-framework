#!/bin/bash
set -x

##### Tested with single node, multiple GPU configuration
### CONFIG ###

DATA_MOUNT=/data/zinc_csv_split
CODE_MOUNT=/workspace/nemo
OUTPUT_MOUNT=/result

JOB_NUM_NODES=1
GPUS_PER_NODE=2

MEGAMOLBART_CONFIG_FILE=small_span_aug
DATA_FILES_SELECTED=x_OP_000..006_CL_.csv

HOSTNAME=LOCALHOST
PROJECT=MegaMolBART

### END CONFIG ###

EXP_NAME=${HOSTNAME}_nodes_${JOB_NUM_NODES}_gpus_${GPUS_PER_NODE}
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}

mkdir -p ${RESULTS_MOUNT}/${EXP_NAME}
GPU_LIMIT=$(($GPUS_PER_NODE-1))
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)

export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

cd ${CODE_MOUNT}/examples/chem
if [ -z ${WANDB_API_KEY} ]; then
    WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
fi

if [ -z ${WANDB_API_KEY} ]; then 
    WANDB_OFFLINE_MODE="true" # handle api key failures gracefully
else
    WANDB_OFFLINE_MODE="false"
fi

python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_${MEGAMOLBART_CONFIG_FILE} \
    dataset_path=${DATA_MOUNT} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${JOB_NUM_NODES} \
    trainer.gpus=${GPUS_PER_NODE} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED}

set +x
