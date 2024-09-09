# BioNeMo2 Repo
To get started, please build the docker container using
```bash
./launch.sh build
```

Launch a container from the build image by executing
```bash
./launch.sh dev
```

All `bionemo2` code is partitioned into independently installable namespace packages. These live under the `sub-packages/` directory.


## Downloading artifacts
Set the AWS access info in your `.env` in the host container prior to running docker:

```bash
AWS_ACCESS_KEY_ID="team-bionemo"
AWS_SECRET_ACCESS_KEY=$(grep aws_secret_access_key ~/.aws/config | cut -d' ' -f 3)
AWS_REGION="us-east-1"
AWS_ENDPOINT_URL="https://pbss.s8k.io"
```
then, running tests should download the test data to a cache location when first invoked.


## Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-fw-ea.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

### Updating pinned versions of NeMo / Megatron-LM

To update the pinned commits of NeMo or Megatron-LM, checkout that commit in the submodule folder, and then commit the
result in the top-level bionemo repository.

```bash
cd 3rdparty/NeMo/
git fetch
git checkout <desired_sha>
cd ../..
git add '3rdparty/NeMo/'
git commit -m "updating NeMo commit"
```


## Testing Locally
Inside the development container, run `./ci/scripts/static_checks.sh` to validate that code changes will pass the code
formatting and license checks run during CI. In addition, run the longer `./ci/scripts/pr_test.sh` script to run unit
tests for all sub-packages.


## Publishing Packages

### Set `BIONEMO_PUBLISH_MODE=1` and `REPO_ROOT`
To publish a bionemo namespace package, you must set the `BIONEMO_PUBLISH_MODE` environment variable to a true value:
one of `"true"`, `"y"`, `"yes"`, or `1` suffices. This will enable the build step to list any dependent bionemo
subpackages by their PyPI-compatible package name. Otherwise, the default is to use local _path_ based dependencies
to reference other bionemo `sub-packages/`.

Additionally, when building, you must manually set the `REPO_ROOT` environment variable to point to the repository's
root on your filesystem. This must be an absolute path. Or, it may be a relative path that starts from the user's home
directory (i.e. `~/` is acceptable).

### Increment `VERSION` File
Make sure that the [`VERSION`](./VERSION) file is incremented appropriately. Bionemo packages follow
[semantic versioning 2.0](https://semver.org/) rules: API-breaking changes are `MAJOR`, new features
are `MINOR`, and bug-fixes and refactors are `PATCH` in `MAJOR.MINOR.PATCH` version string format.

### `python -m build`
Build the bionemo sub-package project by executing the following at the sub-package's root level:
```shell
BIONEMO_PUBLISH_MODE=1 python -m build
```
This will produce a wheel file for the sub-package's code and its dependencies.

### `python -m twine upload`
After building, the wheel file can be uploaded to PyPI (or a compatible package registry) by executing
`python -m twine upload dist/*`.

### All steps together
```bash
echo "Updated version: $(cat VERSION)"
pushd sub-package/bionemo-<project name>
REPO_ROOT="<path to bionemo-fw-ea>" BIONEMO_PUBLISH_MODE=1 python -m build
TWINE_PASSWORD="<pypi pass>" TWINE_USERNAME="<pypi user>" python -m twine upload dist/*
```


## Models
### Geneformer
#### Get test data for geneformer
```bash
mkdir -p /workspace/bionemo2/data
aws s3 cp \
  s3://general-purpose/cellxgene_2023-12-15_small \
  /workspace/bionemo2/data/cellxgene_2023-12-15_small \
  --recursive \
  --endpoint-url https://pbss.s8k.io
```
#### Running

The following command runs a very small example of geneformer:
```bash
TEST_DATA_DIR=$(bionemo_test_data_path single_cell/testdata-20240506 --source pbss); \
python  \
    scripts/singlecell/geneformer/train.py     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data    \
    --result-dir ./results     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2
```

To fine-tune, you just need to specify a different combination of model and loss (TODO also data class). To do that you
pass the path to the config output by the previous step as the `--restore-from-checkpoint-path`, and also change the
`--training-model-config-class` to the new one.

Eventually we will also add CLI options to hot swap in different data modules and processing functions so you could
pass new information into your model for fine-tuning or new targets, but if you want that functionality _now_ you could
copy the `scripts/singlecell/geneformer/train.py` and modify the DataModule class that gets initialized.

Simple fine-tuning example (NOTE: please change `--restore-from-checkpoint-path` to be the one that was output last
by the previous train run)
```bash
TEST_DATA_DIR=$(bionemo_test_data_path single_cell/testdata-20240506 --source pbss); \
python  \
    scripts/singlecell/geneformer/train.py     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data    \
    --result-dir ./results     \
    --experiment-name test_finettune_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --training-model-config-class FineTuneSeqLenBioBertConfig \
    --restore-from-checkpoint-path results/test_experiment/dev/checkpoints/test_experiment--val_loss=10.2042-epoch=0
```


## Updating License Header on Python Files
Make sure you have installed [`license-check`](https://gitlab-master.nvidia.com/clara-discovery/infra-bionemo),
which is defined in the development dependencies. If you add new Python (`.py`) files, be sure to run as:
```bash
license-check --license-header ./license_header --check . --modify --replace
```
