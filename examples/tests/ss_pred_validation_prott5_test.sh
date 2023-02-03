#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####
# Test for ProtT5nv training.
####

### CONFIG ###
PROJECT_MOUNT=${PROJECT_MOUNT:=/workspace/bionemo} # can also be explicity set, e.g. /opt/nvidia/bionemo or /workspace/bionemo
### END CONFIG ###

CONFIG_FILE=prott5nv_sspred_test
DATA_MOUNT=${PROJECT_MOUNT}/examples/tests/test_data/protein
VALID_DATA_MOUNT=${PROJECT_MOUNT}/examples/tests/test_data/protein/downstream
RESULTS_MOUNT=/tmp/results

TRAINING_ARGS="model.data.dataset_path=${DATA_MOUNT}"
TRAINING_ARGS="${TRAINING_ARGS} model.validation.datasets.0.train_ds.data_file=${VALID_DATA_MOUNT}/train/x000.csv"
TRAINING_ARGS="${TRAINING_ARGS} model.validation.datasets.0.test_ds.data_file=${VALID_DATA_MOUNT}/test/x000.csv"

execute() {
    set -x
    python ${PROJECT_MOUNT}/examples/protein/prott5nv/pretrain.py \
        --config-path=${PROJECT_MOUNT}/examples/tests \
        --config-name=${CONFIG_FILE} \
        exp_manager.exp_dir=${RESULTS_MOUNT} \
        ${TRAINING_ARGS} \
        ++exp_manager.create_wandb_logger="False" \
        ++exp_manager.wandb_logger_kwargs.offline="True" >& ${RESULTS_MOUNT}/cmdline_prints
    set +x
}


train() {
    execute
}

rm -rf ${RESULTS_MOUNT}
mkdir -p ${RESULTS_MOUNT}
echo "Setting PROJECT_MOUNT to ${PROJECT_MOUNT}"
export PROJECT_MOUNT=${PROJECT_MOUNT}

if [[ ${PROJECT_MOUNT} != ${BIONEMO_HOME} ]]; then
    echo "Prepending ${PROJECT_MOUNT} to PYTHONPATH for development"
    DEV_PYTHONPATH=${PROJECT_MOUNT}
else
    DEV_PYTHONPATH=""
fi
export PYTHONPATH="${DEV_PYTHONPATH}:${BIONEMO_WORKSPACE}:${BIONEMO_WORKSPACE}/generated:$PYTHONPATH"

export HYDRA_FULL_ERROR=1
cd ${PROJECT_MOUNT}/examples

if [ $# -eq 0 ]; then
    ARGS=train
    CMD='train'
else
    ARGS=$1
    CMD=$@
fi

case $ARGS in
    train)
        $CMD
        ;;
    *)
        usage
        exit 1
        ;;
esac

RES3=$(cat ${RESULTS_MOUNT}/cmdline_prints | grep "Test Data Three State Accuracy") 
arrRes3=(${RES3//,/ })
RES3=${arrRes3[5]}" "${arrRes3[6]}" "${arrRes3[7]}" "${arrRes3[8]}" "${arrRes3[9]}" "${arrRes3[10]:0:5}
RES8=$(cat ${RESULTS_MOUNT}/cmdline_prints | grep "Test Data Eight State Accuracy") 
arrRes8=(${RES8//,/ })
RES8=${arrRes8[5]}" "${arrRes8[6]}" "${arrRes8[7]}" "${arrRes8[8]}" "${arrRes8[9]}" "${arrRes8[10]:0:5} 
echo -e ${RES3}"\n"${RES8} >& ${RESULTS_MOUNT}/result_log

DIFF=$(diff ${RESULTS_MOUNT}/result_log ${PROJECT_MOUNT}/examples/tests/expected_results/prott5nv_sspred_log)
DIFF_PASSED=$?
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
if [ $DIFF_PASSED -ne 0 ]
then
    echo -e "${RED}FAIL${NC}"
    echo "$DIFF"
    cat "${RESULTS_MOUNT}/cmdline_prints"
    exit 1
else
    echo -e "${GREEN}PASS${NC}"
fi
