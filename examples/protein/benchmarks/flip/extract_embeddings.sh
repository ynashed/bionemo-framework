#!/bin/bash
#
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

# This script is used to run inference on a single GPU using the ESM-1b model.

# TODO: use .env variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEVICES=4
DATA_PATH=/data
FLIP_DATA_PATH=${DATA_PATH}/flip
RESULTS_PATH=/result/nemo_experiments/flip

MODEL_NAMES=("esm1nv", "prott5nv")

# embeddings only
DATA_FNAMES_EMBEDDINS_ONLY=(
# aav
    aav/low_vs_high.fasta
    aav/des_mut_nucleotide.fasta
    aav/one_vs_many_nucleotide.fasta
    aav/mut_des_nucleotide.fasta
    aav/seven_vs_many.fasta
    aav/low_vs_high_nucleotide.fasta
    aav/des_mut.fasta
    aav/seven_vs_many_nucleotide.fasta
    aav/two_vs_many_nucleotide.fasta
    aav/mut_des.fasta
    aav/one_vs_many.fasta
    aav/two_vs_many.fasta
# gb1
    gb1/low_vs_high.fasta
    gb1/three_vs_rest.fasta
    gb1/two_vs_rest.fasta
    gb1/one_vs_rest.fasta
    gb1/two_vs_rest_nucleotide.fasta
    gb1/three_vs_rest_nucleotide.fasta
    gb1/low_vs_high_nucleotide.fasta
    gb1/one_vs_rest_nucleotide.fasta
# meltome
    meltome/mixed_split.fasta
    meltome/human_cell.fasta
    meltome/human.fasta
    meltome/mixed_split_nucleotide.fasta
    meltome/human_nucleotide.fasta
    meltome/human_cell_nucleotide.fasta
# sav
    sav/mixed.fasta
)


# hiddens only
DATA_FNAMES_EMBEDDINS_ONLY=(
# secondary_structure
    secondary_structure/sequences.fasta
# bind
    bind/sequences.fasta
# conservation
    conservation/sequences.fasta
)

# embeddings and hiddens
DATA_FNAMES_EMBEDDINS_HIDDENS=(
# scl
    scl/mixed_soft.fasta
    scl/mixed_hard.fasta
)

# extract embeddings
for DATA_FNAME in ${DATA_FNAMES_EMBEDDINS_ONLY}; do
    DATA_FILE=${FLIP_DATA_PATH}/${DATA_FNAME}
    for MODEL_NAME in ${MODEL_NAMES}; do
        echo "\n**********************************************************"
        echo "Extracting embeddings for ${DATA_FNAME} with ${MODEL_NAME}"
        echo "**********************************************************\n"

        ${SCRIPT_DIR}/../../${MODEL_NAME}/infer.sh \
            trainer.devices=${DEVICES} \
            model.data.dataset_path=${DATA_FILE} \
            model.data.output_fname=${DATA_FILE}.${MODEL_NAME}.pkl
    done
done
