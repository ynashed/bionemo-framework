# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import scipy

from bionemo.data.scdl.api.single_cell_row_dataset import SingleCellRowDataset
from bionemo.data.scdl.index.feature_index import RowFeatureIndex
from bionemo.data.scdl.util.string_enum import StringEnum


class ArrNames(StringEnum):
    DATA = "data.npy"
    COLPTR = "col_ptr.npy"
    ROWPTR = "row_ptr.npy"
    VERSION = "version.json"
    METADATA = "metadata.json"
    DTYPE = "dtypes.json"
    FEATURES = "features.idx"


class Mode(StringEnum):
    """Valid modes for the single cell memory mapped dataset: either write or read append.

    The write append mode is 'w+' while the read append mode is 'r+'.
    """

    CREATE_APPEND = "w+"
    READ_APPEND = "r+"
    READ = "r"


class METADATA(StringEnum):
    NUM_ROWS = "num_rows"


def _swap_mmap_array(
    src_array: np.memmap,
    src_path: str,
    dest_array: np.memmap,
    dest_path: str,
    destroy_src: bool = False,
) -> None:
    # TODO: docstring
    assert os.path.isfile(src_path)
    assert os.path.isfile(dest_path)

    # Flush and close arrays
    src_array.flush()
    dest_array.flush()

    del src_array
    del dest_array

    # Swap the file locations on disk using a tmp file.
    with tempfile.TemporaryDirectory() as tempdir:
        temp_file_name = f"{tempdir}/arr_temp"
        os.rename(src_path, temp_file_name)
        os.rename(dest_path, src_path)
        os.rename(temp_file_name, dest_path)

    if destroy_src:
        os.remove(src_path)


def _pad_sparse_array(row_values, row_col_ptr, n_cols: int) -> np.ndarray:
    ret = np.zeros(n_cols)
    for i in range(0, len(row_values)):
        col = row_col_ptr[i]
        ret[col] = row_values[i]
    return ret


def _create_arrs(
    n_elements: int,
    n_rows: int,
    path: Path,
    mode: Mode,
    dtypes: Dict[str, str],
    create_path_if_nonexistent: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the parameters required to create
    a set of CSR-format numpy arrays and
    creates them at path.
    Nice way of indexing into a sparse matrix.
    Data stores the non-zero values within the matrix.
    col - stores column index for a given non-zero value (1d array length of the dataset?)
    row - stores a pointer to slice into the rows of the dataset. value at row_id - 1 is index of first element of
    the row in data and column array
    """
    assert ArrNames.DATA in dtypes
    assert ArrNames.COLPTR in dtypes
    assert ArrNames.ROWPTR in dtypes

    assert n_elements > 0
    assert n_rows > 0

    if not create_path_if_nonexistent and not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    # mmap new arrays
    # Records the value at index[i]
    data_arr = np.memmap(f"{path}/{ArrNames.DATA}", dtype=dtypes[ArrNames.DATA], shape=(n_elements,), mode=mode)
    # Records the column the data resides in at index [i]
    col_arr = np.memmap(f"{path}/{ArrNames.COLPTR}", dtype=dtypes[ArrNames.COLPTR], shape=(n_elements,), mode=mode)
    # Records a pointer into the data and column arrays
    # to get the data for a specific row, slice row_idx[idx, idx+1]
    # and then get the elements in data[row_idx[idx]:row_idx[idx+1]]
    # which are in the corresponding columns col_index[row_idx[idx], row_idx[row_idx+1]]
    row_arr = np.memmap(f"{path}/{ArrNames.ROWPTR}", dtype=dtypes[ArrNames.ROWPTR], shape=(n_rows + 1,), mode=mode)
    return data_arr, col_arr, row_arr


class SingleCellMemMapDataset(SingleCellRowDataset):
    def __init__(
        self,
        data_path: str,
        h5ad_path: Optional[str] = None,
        n_values: Optional[int] = None,
        n_obs: Optional[int] = None,
        mode: Mode = Mode.READ_APPEND,
    ) -> None:
        """
        Loads an existing SCMMAP Dataset or Creates a new SingleCellMemMapDataset from an H5AD file.
        data_path
        h5ad_path
        """

        self._version: str = "0.0.1"
        self.data_path: str = data_path
        self.mode: Mode = mode

        # Backing arrays
        self.data: Optional[np.ndarray] = None
        self.row_index: Optional[np.ndarray] = None
        self.col_index: Optional[np.ndarray] = None

        # Metadata and attributes
        self.metadata: Dict = {}

        # Stores the Feature Index, which tracks
        # the original AnnData features (e.g., gene names)
        # and allows us to store ragged arrays in our SCMMAP structure.
        self._feature_index: RowFeatureIndex = RowFeatureIndex()

        # Variables for int packing / reduced precision
        self.dtypes: Dict[ArrNames, str] = {
            ArrNames.DATA: "float32",
            ArrNames.COLPTR: "uint32",
            ArrNames.ROWPTR: "uint64",
        }

        if mode == Mode.CREATE_APPEND and os.path.exists(data_path):
            raise FileExistsError(f"Output directory already exists: {data_path}")

        if h5ad_path is not None and (data_path is not None and os.path.exists(data_path)):
            raise FileExistsError(
                "Invalid input; both an existing SCMMAP and an h5ad file were passed. "
                "Please pass either an existing SCMMAP or an h5ad file."
            )

        # If there is only a data path, and it exists already, load SCMMAP data.
        elif data_path is not None and os.path.exists(data_path):
            self.__init__obj()
            self.load(data_path)

        # If there is only an h5ad path, load the HDF5 data
        elif h5ad_path is not None:
            self.__init__obj()
            self.load_h5ad(h5ad_path, "float32")

        elif isinstance(n_obs, int) and isinstance(n_values, int):
            self.__init__obj()
            self._init_arrs(n_elements=n_values, n_rows=n_obs)

    def __init__obj(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{ArrNames.VERSION}"):
            with open(f"{self.data_path}/{ArrNames.VERSION}", "w") as vfi:
                json.dump(self.version(), vfi)

    def _init_arrs(self, n_elements: int, n_rows: int) -> None:
        self.mode = Mode.CREATE_APPEND
        data_arr, col_arr, row_arr = _create_arrs(
            n_elements, n_rows, Path(self.data_path), self.mode, self.dtypes, True
        )
        self.data = data_arr
        self.col_index = col_arr
        self.row_index = row_arr

    def _get_row(
        self,
        idx,
        return_features: bool = False,
        pad: bool = False,
        feature_vars: Optional[List[str]] = None,
    ):  # TODO: sort this shit out
        """
        Union[
            Tuple[np.ndarray, np.ndarray],
            np.ndarray,
            Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame],
            Tuple[np.ndarray, pd.DataFrame],
        ]:"""
        start = self.row_index[idx]
        end = self.row_index[idx + 1]
        values = self.data[start:end]
        columns = self.col_index[start:end]
        ret = (values, columns)

        if pad:
            ret = _pad_sparse_array(values, columns, self._feature_index.n_vars_at_row(idx))
        if return_features:
            return ret, self._feature_index.lookup(idx, select_features=feature_vars)[0]
        return ret, None

    def get(
        self, row: int, column: Optional[int] = None, impute_missing_zeros: bool = True, pad: bool = True
    ) -> Optional[float | np.ndarray | tuple[np.ndarray, np.ndarray]]:
        """
        Obtain the entry at the specified row and optionally column.
        TODO: REALLY SORT THIS
        """
        (row_values, row_col_ptr), _ = self._get_row(row, pad=False)

        if column is not None:
            for i, col in enumerate(row_col_ptr):
                if col == column:
                    # return the value at this position
                    return row_values[i]
                elif col > column:
                    try:
                        raise ValueError(f"Column pointer {col} exceeds the number of columns {column} for the row.")
                    except ValueError:
                        break
            return 0.0 if impute_missing_zeros else None

        if pad:
            return _pad_sparse_array(row_values, self._feature_index.n_vars_at_row(row), row_col_ptr)
        else:
            return row_values, row_col_ptr

    def version(self) -> str:
        return self._version

    def features(self) -> Optional[RowFeatureIndex]:
        return self._feature_index

    def _load_mmap_file_if_exists(self, file_path, dtype):
        if os.path.exists(file_path):
            return np.memmap(file_path, dtype=dtype, mode=self.mode)
        else:
            raise FileNotFoundError(f"The mmap file at {file_path} is missing")

    def load(self, stored_path: str) -> None:
        if not os.path.exists(stored_path):
            # TODO, what do for instructions here
            raise FileNotFoundError(f"Error: could not find data path [{stored_path}]")
        self.data_path = stored_path
        self.mode = Mode.READ_APPEND

        # Read the version
        with open(f"{self.data_path}/{ArrNames.VERSION}", Mode.READ) as vfi:
            self.version = json.load(vfi)

        # Metadata is required, so we must check if it exists and fail if not.
        assert os.path.exists(f"{self.data_path}/{ArrNames.METADATA}")
        with open(f"{self.data_path}/{ArrNames.METADATA}", Mode.READ) as mfi:
            self.metadata = json.load(mfi)

        self._feature_index = RowFeatureIndex.load(f"{self.data_path}/{ArrNames.FEATURES}")

        # DTYPE is not required, though maybe it should be.
        # TODO: this should also include the INTWIDTH array(s)
        if os.path.exists(f"{self.data_path}/{ArrNames.DTYPE}"):
            with open(f"{self.data_path}/{ArrNames.DTYPE}") as dfi:
                self.dtypes = json.load(dfi)

        # TODO: deprecated
        # Features, which holds gene names / ENSEMBL IDs in some form.
        # Unfortunately, there are __zero__ gurantees about the structure of this except
        # that it will have the same number of rows as the dataset has columns.
        # it should probably be refactored into another class at some point.
        # if os.path.exists(f"{self.data_path}/{ArrNames.FEATURES}"):
        #     self._features = read_feather(f"{self.data_path}/{ArrNames.FEATURES}")

        # mmap the existing arrays
        self.data = self._load_mmap_file_if_exists(f"{self.data_path}/{ArrNames.DATA}", self.dtypes[ArrNames.DATA])
        self.row_index = self._load_mmap_file_if_exists(
            f"{self.data_path}/{ArrNames.ROWPTR}", dtype=self.dtypes[ArrNames.ROWPTR]
        )
        self.col_index = self._load_mmap_file_if_exists(
            f"{self.data_path}/{ArrNames.COLPTR}", dtype=self.dtypes[ArrNames.COLPTR]
        )

        # Set variables based on arrays,
        # checking that they match the metadata if it's present.

    def _write_metadata(self) -> None:
        with open(f"{self.data_path}/{ArrNames.METADATA}", "w") as mfi:
            json.dump(self.metadata, mfi)

    def load_h5ad(
        self,
        anndata_path: str,
        dtype: str = "int32",
    ) -> None:
        """
        Takes a path to an existing AnnData archive,
        loads the data from disk and then creates a new backing data structure.
        Note: the storage utilized will roughly double.
        """
        if not os.path.exists(anndata_path):
            raise FileNotFoundError(f"Error: could not find h5ad path [{anndata_path}]")
        adata = ad.read_h5ad(anndata_path)  # slow
        # Get / set the number of rows and columns for sanity
        # Fill the data array
        # TODO: these are not universal solutions
        # They will definitely work for scipy::csr_matrices, but we may
        # need to implement special logic for e.g., ndarrays.

        if not isinstance(adata.X, scipy.sparse.spmatrix):
            raise NotImplementedError("Error: dense matrix loading not yet implemented.")

        # TODO: impl getattr(data, "raw", None)

        # Check if raw data is present
        raw = getattr(adata, "raw", None)
        if raw is not None:
            # If it is, attempt to get the counts in the raw data.
            count_data = getattr(raw, "X", None)

        if count_data is None:
            # No raw counts were present, resort to normalized
            count_data = getattr(adata, "X")
        if count_data is None:
            raise ValueError("This file does not have count data")

        shape = count_data.shape
        num_rows = shape[0]

        n_stored = count_data.nnz
        self.dtypes[ArrNames.DATA] = count_data.dtype

        # Collect features and store in FeatureIndex
        features = adata.var
        self._feature_index.append_features(n_obs=num_rows, features=features, label=anndata_path)

        # Create the arrays.
        self._init_arrs(n_stored, num_rows)
        # Store data
        self.data[0:n_stored] = count_data.data

        # Store the col idx array
        self.col_index[0:n_stored] = count_data.indices.astype(int)

        # Store the row idx array
        self.row_index[0 : num_rows + 1] = count_data.indptr.astype(int)

        self.save()

    def save(self, output_path: Optional[str] = None) -> None:
        assert isinstance(self.n_obs(), int)
        if METADATA.NUM_ROWS not in self.metadata:
            self.metadata[METADATA.NUM_ROWS] = self.n_obs()

        self._write_metadata()
        # Write the feature index.
        self._feature_index.save(f"{self.data_path}/{ArrNames.FEATURES}")
        # Ensure the object is in a valid state.
        for postfix in [ArrNames.VERSION, ArrNames.DATA, ArrNames.COLPTR, ArrNames.ROWPTR, ArrNames.FEATURES]:
            if not os.path.exists(f"{self.data_path}/{postfix}"):
                raise FileNotFoundError(f"Path does not exist: {self.data_path}/{postfix}")

        self.data.flush()
        self.row_index.flush()
        self.col_index.flush()

        if output_path is not None:
            raise NotImplementedError("Saving to separate path is not yet implemented.")

        # TODO: add compression phase here.
        return True

    def n_values(self):
        return sum(self._feature_index.n_values())

    def n_obs(self):
        return self._feature_index.n_rows()

    def __len__(self):
        """
        Return the number of observations.
        """
        return self.n_obs()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:  # TODO: check this
        return self._get_row(idx, pad=False, feature_vars=None)[0]

    def n_vars(self) -> List[int]:
        feats = self._feature_index
        if len(feats) == 0:
            return [0]
        num_vars = feats.column_dims()
        return num_vars

    def num_nonzeros(self) -> int:
        return self.data.size

    def shape(self) -> Tuple[int, List[int]]:
        return self.n_obs(), self.n_vars()

    def compare_features(self, other, cols_to_compare: Tuple[str] = ("feature_name")) -> int:
        """
        Returns True if features specified in cols_to_compare are identical, false otherwise.
        """
        feats_self = self._features
        feats_other = other._features
        col_identities = [False] * len(cols_to_compare)  # track identity for each column

        # TODO: could implement early-exit
        def iden(x, y):
            return x.equals(y)

        col_identities = [iden(feats_self[c], feats_other[c]) for c in cols_to_compare]
        return all(col_identities)

    def concat(
        self,
        other_dataset: Union[list["SingleCellMemMapDataset"], "SingleCellMemMapDataset"],
    ) -> None:
        """
        Takes another SCMMAP class and concatenates it to the existing one.
        """
        # Verify the versions are compatible.
        if isinstance(other_dataset, type(self)):
            assert self.version() == other_dataset.version()
        elif isinstance(other_dataset, list):
            for o in other_dataset:
                assert self.version() == o.version()
        else:
            raise ValueError(
                f"Expecting either a {SingleCellMemMapDataset} or a list thereof. Actually got: {type(other_dataset)}"
            )

        # Set our mode:
        self.mode: Mode = Mode.READ_APPEND

        # TODO: Add to our metadata
        mmaps = []
        if isinstance(other_dataset, list):
            mmaps.extend(other_dataset)
        else:
            mmaps.append(other_dataset)
        # Calculate the size of our new dataset arrays
        total_n_elements = (self.num_nonzeros() if self.n_obs() > 0 else 0) + sum([m.num_nonzeros() for m in mmaps])
        total_n_rows = self.n_obs() + sum([m.n_obs() for m in mmaps])

        # Create new arrays to store the data, colptr, and rowptr.
        with tempfile.TemporaryDirectory(prefix="_tmp", dir=self.data_path) as tmp:
            # TODO: ensure proper DTYPE compression / overflow protection
            data_arr, col_arr, row_arr = _create_arrs(
                n_elements=total_n_elements,
                n_rows=total_n_rows,
                path=Path(tmp),
                mode=Mode.CREATE_APPEND,
                dtypes=self.dtypes,  # TODO: make safe for overflow.
                create_path_if_nonexistent=True,
            )
            # Copy the data from self and other into the new arrays.
            cumulative_elements = 0
            cumulative_rows = 0
            if self.n_obs() > 0:
                data_arr[cumulative_elements : cumulative_elements + self.num_nonzeros()] = self.data.data
                col_arr[cumulative_elements : cumulative_elements + self.num_nonzeros()] = self.col_index.data
                row_arr[cumulative_rows : cumulative_rows + self.n_obs() + 1] = self.row_index.data
                cumulative_elements += self.num_nonzeros()
                cumulative_rows += self.n_obs()
            for mmap in mmaps:
                # Fill the data array for the span of this scmmap
                data_arr[cumulative_elements : cumulative_elements + mmap.num_nonzeros()] = mmap.data.data
                # fill the col array for the span of this scmmap
                col_arr[cumulative_elements : cumulative_elements + mmap.num_nonzeros()] = mmap.col_index.data
                # Fill the row array for the span of this scmmap
                row_arr[cumulative_rows : cumulative_rows + mmap.n_obs() + 1] = (
                    mmap.row_index + int(cumulative_rows)
                ).data

                self._feature_index.concat(mmap._feature_index)
                # Update counters
                cumulative_elements += mmap.num_nonzeros()
                cumulative_rows += mmap.n_obs()
            # TODO: Replace self's arrays with the newly filled arrays.
            _swap_mmap_array(
                data_arr, f"{tmp}/{ArrNames.DATA}", self.data, f"{self.data_path}/{ArrNames.DATA}", destroy_src=True
            )
            _swap_mmap_array(
                col_arr,
                f"{tmp}/{ArrNames.COLPTR}",
                self.col_index,
                f"{self.data_path}/{ArrNames.COLPTR}",
                destroy_src=True,
            )
            _swap_mmap_array(
                row_arr,
                f"{tmp}/{ArrNames.ROWPTR}",
                self.row_index,
                f"{self.data_path}/{ArrNames.ROWPTR}",
                destroy_src=True,
            )
            # Reopen the data, colptr, and rowptr arrays
            self.data = np.memmap(
                f"{self.data_path}/{ArrNames.DATA}",
                dtype=self.dtypes[ArrNames.DATA],
                shape=(cumulative_elements,),
                mode=Mode.READ_APPEND,
            )
            self.row_index = np.memmap(
                f"{self.data_path}/{ArrNames.ROWPTR}",
                dtype=self.dtypes[ArrNames.ROWPTR],
                shape=(cumulative_rows + 1,),
                mode=Mode.READ_APPEND,
            )
            self.col_index = np.memmap(
                f"{self.data_path}/{ArrNames.COLPTR}",
                dtype=self.dtypes[ArrNames.COLPTR],
                shape=(cumulative_elements,),
                mode=Mode.READ_APPEND,
            )
            # TODO: Verify mmap integrity upon file move.

        # Update the number of rows and number of columns
        # self.num_rows = cumulative_rows
        # TODO: Validate the feature index integrity
        # TODO: remove the num_columns index
        self.save()
