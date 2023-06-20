#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

source artifact_paths

ngc_api_key_is_set() {
    if [ ! -z ${NGC_CLI_API_KEY} ] && [ ${NGC_CLI_API_KEY} != 'NotSpecified' ]; then
        echo true
    else
        echo false
    fi
}


function setup_model() {
    local model_source=$1
    local base_directory=$2
    local model_target=${base_directory}/$3

    set -e

    echo "Downloading model ${model_source} to ${model_target}..."
    local tmp_root=`mktemp -d`
    local tmp_download_loc="${tmp_root}/bionemo_downloads"
    rm -rf ${tmp_download_loc}
    mkdir -p ${tmp_download_loc}

    if [[ ${model_source} = http* ]]; then
        wget -q --show-progress ${model_source} -O ${tmp_download_loc}/model.zip
        download_path=$(unzip -o ${tmp_download_loc}/model.zip -d ${base_directory} | grep "inflating:")
        download_path=$(echo ${download_path} | cut -d ":" -f 2)
        model_basename=$(basename ${download_path})
        downloaded_model_file="${base_directory}/${model_basename}"
        rm -rf ${tmp_download_loc}/model.zip
    else
        ngc registry model download-version \
            --dest ${tmp_download_loc} \
            "${model_source}"

        download_path=$(ls ${tmp_download_loc}/*/*.nemo)
        model_basename=$(basename ${download_path%.nemo})
        mv ${download_path} ${base_directory}
        downloaded_model_file="${base_directory}/${model_basename}.nemo"
        rm -rf ${tmp_download_loc}/*
    fi

    echo "Linking ${downloaded_model_file} to ${model_target}..."
    mkdir -p $(dirname ${model_target})
    ln -frs ${downloaded_model_file} ${model_target}
    
    # This file is created to record the version of model
    mkdir -p ${base_directory}/version
    touch ${base_directory}/version/${model_source//[\/]/_}.version
    set +e
}


download_bionemo_models() {
    local ngc_api_key_is_set_=$(ngc_api_key_is_set)
    if [ $ngc_api_key_is_set_ != true ]; then
        echo 'The NGC cli key ($NGC_CLI_API_KEY) is not set correctly. Model download may fail.'
    fi

    mkdir -p ${MODEL_PATH}
    setup_model \
        "${MEGAMOLBART_MODEL}" \
        "${MODEL_PATH}" \
        "molecule/megamolbart/megamolbart.nemo"
    setup_model \
        "${ESM1NV_MODEL}" \
        "${MODEL_PATH}" \
        "protein/esm1nv/esm1nv.nemo"
    setup_model \
        "${PROTT5NV_MODEL}" \
        "${MODEL_PATH}" \
        "protein/prott5nv/prott5nv.nemo"
}
