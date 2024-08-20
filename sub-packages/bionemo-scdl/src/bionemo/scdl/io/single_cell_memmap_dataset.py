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


import importlib.metadata
import json
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import torch

from bionemo.scdl.api.single_cell_row_dataset import SingleCellRowDataset
from bionemo.scdl.index.row_feature_index import RowFeatureIndex


class FileNames(str, Enum):
    """Names of files that are generated in SingleCellCollection."""

    DATA = "data.npy"
    COLPTR = "col_ptr.npy"
    ROWPTR = "row_ptr.npy"
    METADATA = "metadata.json"
    DTYPE = "dtypes.json"
    FEATURES = "features"
    VERSION = "version.json"


class Mode(str, Enum):
    """Valid modes for the single cell memory mapped dataset.

    The write append mode is 'w+' while the read append mode is 'r+'.
    """

    CREATE_APPEND = "w+"
    READ_APPEND = "r+"
    READ = "r"
    CREATE = "w"


class METADATA(str, Enum):
    """Stored metadata."""

    NUM_ROWS = "num_rows"


def _swap_mmap_array(
    src_array: np.memmap,
    src_path: str,
    dest_array: np.memmap,
    dest_path: str,
    destroy_src: bool = False,
) -> None:
    """Function that swaps the location of two mmap arrays.

    This is used when concatanating SingleCellMemMapDataset. This emables the
    newly merged arrays to be stored in the same place as the original dataset.

    Args:
        src_array: the first memmap array
        src_path: location of the first memmap array
        dest_array: the second memmap array
        dest_path: location of the first second array
        destroy_src: set to True if the source array is destroyed

    Raises:
        FileNotFoundError if the source or destination path are not found.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"The source file {src_path} does not exist")
    if not os.path.isfile(dest_path):
        raise FileNotFoundError(f"The destination file {dest_path} does not exist")

    # Flush and close arrays
    src_array.flush()
    dest_array.flush()

    del src_array
    del dest_array

    # Swap the file locations on disk using a tmp file.
    with tempfile.TemporaryDirectory() as tempdir:
        temp_file_name = f"{tempdir}/arr_temp"
        shutil.move(src_path, temp_file_name)
        shutil.move(dest_path, src_path)
        shutil.move(temp_file_name, dest_path)

    if destroy_src:
        os.remove(src_path)


def _pad_sparse_array(row_values, row_col_ptr, n_cols: int) -> np.ndarray:
    """Creates a conventional array from a sparse one.

    Convert a sparse matrix representation of a 1d matrix to a conventional
    numpy representation.

    Args:
        row_values: The row indices of the entries
        row_col_ptr: The corresponding column pointers
        n_cols: The number of columns in the dataset.

    Returns:
        The full 1d numpy array representation.
    """
    ret = np.zeros(n_cols)
    for row_ptr in range(0, len(row_values)):
        col = row_col_ptr[row_ptr]
        ret[col] = row_values[row_ptr]
    return ret


def _create_compressed_sparse_row_memmaps(
    num_elements: int,
    num_rows: int,
    memmap_dir_path: Path,
    mode: Mode,
    dtypes: Dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a set of CSR-format numpy arrays.

    They are saved to memmap_dir_path. This is an efficient way of indexing
    into a sparse matrix. Only non- zero values of the data are stored.
    """
    if num_elements <= 0:
        raise ValueError(f"n_elements is set to {num_elements}. It must be postive to create CSR matrices.")

    if num_rows <= 0:
        raise ValueError(f"num_rows is set to {num_rows}. It must be postive to create CSR matrices.")

    memmap_dir_path.mkdir(parents=True, exist_ok=True)
    # mmap new arrays
    # Records the value at index[i]
    data_arr = np.memmap(
        f"{memmap_dir_path}/{FileNames.DATA}", dtype=dtypes[f"{FileNames.DATA}"], shape=(num_elements,), mode=mode
    )
    # Records the column the data resides in at index [i]
    col_arr = np.memmap(
        f"{memmap_dir_path}/{FileNames.COLPTR}", dtype=dtypes[f"{FileNames.COLPTR}"], shape=(num_elements,), mode=mode
    )
    # Records a pointer into the data and column arrays
    # to get the data for a specific row, slice row_idx[idx, idx+1]
    # and then get the elements in data[row_idx[idx]:row_idx[idx+1]]
    # which are in the corresponding columns col_index[row_idx[idx], row_idx[row_idx+1]]
    row_arr = np.memmap(
        f"{memmap_dir_path}/{FileNames.ROWPTR}", dtype=dtypes[f"{FileNames.ROWPTR}"], shape=(num_rows + 1,), mode=mode
    )
    return data_arr, col_arr, row_arr


class SingleCellMemMapDataset(SingleCellRowDataset):
    """Represents one or more AnnData matrices.

    Data is stored in large, memory-mapped arrays that enables fast access of
    datasets larger than the available amount of RAM on a system. SCMMAP
    implements a consistent API defined in SingleCellRowDataset.

    Attributes:
        data_path: Location of np.memmap files to be loaded from or that will be
        created.
        mode: Whether the dataset will be read in (r+) from np.memmap files or
        written to np.memmap files (w+).
        data: A numpy array of the data
        row_index: A numpy array of row pointers
        col_index: A numpy array of column values
        metadata: Various metata about the dataset.
        _feature_index: The corresponding RowFeatureIndex where features are
        stored
        dtypes: A dictionary containing the datatypes of the data, row_index,
        and col_index arrays.
        _version: The version of the dataset
    """

    def __init__(
        self,
        data_path: str,
        h5ad_path: Optional[str] = None,
        num_elements: Optional[int] = None,
        num_rows: Optional[int] = None,
        mode: Mode = f"{Mode.READ_APPEND}",
    ) -> None:
        """Instantiate the class.

        Args:
            data_path: The location where the data np.memmap files are read from
            or stored.
            h5ad_path: Optional, the location of the h5_ad path.
            num_elements: The total number of elements in the array.
            num_rows: The number of rows in the data frame
            mode: Whether to read or write from the data_path,
        """
        self._version: str = importlib.metadata.version("bionemo.scdl")
        self.data_path: str = data_path
        self.mode: Mode = mode

        # Backing arrays
        self.data: Optional[np.ndarray] = None
        self.row_index: Optional[np.ndarray] = None
        self.row_index: Optional[np.ndarray] = None

        # Metadata and attributes
        self.metadata: Dict[str, int] = {}

        # Stores the Feature Index, which tracks
        # the original AnnData features (e.g., gene names)
        # and allows us to store ragged arrays in our SCMMAP structure.
        self._feature_index: RowFeatureIndex = RowFeatureIndex()

        # Variables for int packing / reduced precision
        self.dtypes: Dict[FileNames, str] = {
            f"{FileNames.DATA}": "float32",
            f"{FileNames.COLPTR}": "uint32",
            f"{FileNames.ROWPTR}": "uint64",
        }

        if mode == f"{Mode.CREATE_APPEND}" and os.path.exists(data_path):
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
            self.load_h5ad(h5ad_path)
        else:
            match num_rows, num_elements:
                case (int(), int()):
                    self.__init__obj()
                    self._init_arrs(num_elements=num_elements, num_rows=num_rows)
                case _:
                    raise ValueError(
                        "An np.memmap path, an h5ad path, or the number of elements and rows is required" ""
                    )

    def __init__obj(self):
        """Initializes the datapath and writes the version."""
        os.makedirs(self.data_path, exist_ok=True)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{FileNames.VERSION}"):
            with open(f"{self.data_path}/{FileNames.VERSION}", "w") as vfi:
                json.dump(self.version(), vfi)

    def _init_arrs(self, num_elements: int, num_rows: int) -> None:
        self.mode = f"{Mode.CREATE_APPEND}"
        data_arr, col_arr, row_arr = _create_compressed_sparse_row_memmaps(
            num_elements=num_elements,
            num_rows=num_rows,
            memmap_dir_path=Path(self.data_path),
            mode=self.mode,
            dtypes=self.dtypes,
        )
        self.data = data_arr
        self.col_index = col_arr
        self.row_index = row_arr

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def get_row(
        self,
        index: int,
        return_features: bool = False,
        feature_vars: Optional[List[str]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
        """Returns a given row in the dataset along with optional features.

        Args:
            index: The row to be returned. This is in the range of [0, num_rows)
            return_features: boolean that indicates whether to return features
            feature_vars: Optional, feature variables to extract
        Return:
            [Tuple[np.ndarray, np.ndarray]: data values and column pointes
            pd.DataFrame: optional, corresponding features.
        """
        start = self.row_index[index]
        end = self.row_index[index + 1]
        values = self.data[start:end]
        columns = self.col_index[start:end]
        ret = (values, columns)
        if return_features:
            return ret, self._feature_index.lookup(index, select_features=feature_vars)[0]
        else:
            return ret, None

    def get_row_padded(
        self,
        index: int,
        return_features: bool = False,
        feature_vars: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Returns a padded version of a row in the dataset.

        A padded version is one where the a sparse array representation is
        converted to a conventional represenentation. Optionally, features are
        returned.

        Args:
            index: The row to be returned
            return_features: boolean that indicates whether to return features
            feature_vars: Optional, feature variables to extract
        Return:
            np.ndarray: conventional row representation
            pd.DataFrame: optional, corresponding features.
        """
        (row_values, row_column_pointer), features = self.get_row(index, return_features, feature_vars)
        return (
            _pad_sparse_array(row_values, row_column_pointer, self._feature_index.number_vars_at_row(index)),
            features,
        )

    def get_row_column(self, index: int, column: int, impute_missing_zeros: bool = True) -> Optional[float]:
        """Returns the value at a given index and the corresponding column.

        Args:
            index: The index to be returned
            column: The column to be returned
            impute_missing_zeros: boolean that indicates whether to set missing
            data to 0
        Return:
            A float that is the value in the array or None.
        """
        (row_values, row_column_pointer), _ = self.get_row(index)
        if column is not None:
            for col_index, col in enumerate(row_column_pointer):
                if col == column:
                    # return the value at this position
                    return row_values[col_index]
                elif col > column:
                    try:
                        raise ValueError(f"Column pointer {col} is larger than the column {column}.")
                    except ValueError:
                        break
            return 0.0 if impute_missing_zeros else None

    def features(self) -> Optional[RowFeatureIndex]:
        """Return the corresponding RowFeatureIndex."""
        return self._feature_index

    def _load_mmap_file_if_exists(self, file_path, dtype):
        if os.path.exists(file_path):
            return np.memmap(file_path, dtype=dtype, mode=self.mode)
        else:
            raise FileNotFoundError(f"The mmap file at {file_path} is missing")

    def load(self, stored_path: str) -> None:
        """Loads the data at store_path that is an np.memmap format.

        Args:
            stored_path: directory with np.memmap files
        Raises:
            FileNotFoundError if the corresponding directory or files are not
            found, or if the metadata file is not present.
        """
        if not os.path.exists(stored_path):
            raise FileNotFoundError(
                f"""Error: the specified data path to the mmap files {stored_path} does not exist.
                                    Specify an updated filepath or provide an h5ad path to the dataset. The data can
                                    be loaded with SingleCellMemMapDataset.load_h5ad. Alternatively, the class can be instantiated
                                    with  SingleCellMemMapDataset(<path to data that will be created>, h5ad_path=<path to h5ad file>"""
            )
        self.data_path = stored_path
        self.mode = f"{Mode.READ_APPEND}"

        # Metadata is required, so we must check if it exists and fail if not.
        if not os.path.exists(f"{self.data_path}/{FileNames.METADATA}"):
            raise FileNotFoundError(f"Error: the metadata file {self.data_path}/{FileNames.METADATA} does not exist.")

        with open(f"{self.data_path}/{FileNames.METADATA}", f"{Mode.READ}") as mfi:
            self.metadata = json.load(mfi)

        if os.path.exists(f"{self.data_path}/{FileNames.FEATURES}"):
            self._feature_index = RowFeatureIndex.load(f"{self.data_path}/{FileNames.FEATURES}")

        if os.path.exists(f"{self.data_path}/{FileNames.DTYPE}"):
            with open(f"{self.data_path}/{FileNames.DTYPE}") as dfi:
                self.dtypes = json.load(dfi)

        # mmap the existing arrays
        self.data = self._load_mmap_file_if_exists(
            f"{self.data_path}/{FileNames.DATA}", self.dtypes[f"{FileNames.DATA}"]
        )
        self.row_index = self._load_mmap_file_if_exists(
            f"{self.data_path}/{FileNames.ROWPTR}", dtype=self.dtypes[f"{FileNames.ROWPTR}"]
        )
        self.col_index = self._load_mmap_file_if_exists(
            f"{self.data_path}/{FileNames.COLPTR}", dtype=self.dtypes[f"{FileNames.COLPTR}"]
        )

    def _write_metadata(self) -> None:
        with open(f"{self.data_path}/{FileNames.METADATA}", f"{Mode.CREATE}") as mfi:
            json.dump(self.metadata, mfi)

    def load_h5ad(
        self,
        anndata_path: str,
    ) -> None:
        """Loads an existing AnnData archive from disk.

        This creates a new backing data structure which is saved.
        Note: the storage utilized will roughly double. Currently, the data must
        be in a scipy.sparse.spmatrix format.

        Args:
            anndata_path: location of data load
        Raises:
            FileNotFoundError if the data path does not exist.
            NotImplementedError if the data is not in scipy.sparse.spmatrix
            format
            ValueError it there is not count data
        """
        if not os.path.exists(anndata_path):
            raise FileNotFoundError(f"Error: could not find h5ad path {anndata_path}")
        adata = ad.read_h5ad(anndata_path)  # slow
        # Get / set the number of rows and columns for sanity
        # Fill the data array
        if not isinstance(adata.X, scipy.sparse.spmatrix):
            raise NotImplementedError("Error: dense matrix loading not yet implemented.")

        # Check if raw data is present
        raw = getattr(adata, "raw", None)
        count_data = None
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

        num_elements_stored = count_data.nnz
        self.dtypes[f"{FileNames.DATA}"] = count_data.dtype

        # Collect features and store in FeatureIndex
        features = adata.var
        self._feature_index.append_features(n_obs=num_rows, features=features, label=anndata_path)

        # Create the arrays.
        self._init_arrs(num_elements_stored, num_rows)
        # Store data
        self.data[0:num_elements_stored] = count_data.data

        # Store the col idx array
        self.col_index[0:num_elements_stored] = count_data.indices.astype(int)

        # Store the row idx array
        self.row_index[0 : num_rows + 1] = count_data.indptr.astype(int)

        self.save()

    def save(self, output_path: Optional[str] = None) -> None:
        """Saves the class to a given output path.

        Args:
            output_path: The location to save - not yet implemented and should
            be self.data_path

        Raises:
           NotImplementedError if output_path is not None.
        """
        if f"{METADATA.NUM_ROWS}" not in self.metadata:
            self.metadata[f"{METADATA.NUM_ROWS}"] = self.number_of_rows()

        self._write_metadata()
        # Write the feature index. This may not exist.
        self._feature_index.save(f"{self.data_path}/{FileNames.FEATURES}")

        # Ensure the object is in a valid state. These are saved at creation!
        for postfix in [
            f"{FileNames.VERSION}",
            f"{FileNames.DATA}",
            f"{FileNames.COLPTR}",
            f"{FileNames.ROWPTR}",
            f"{FileNames.FEATURES}",
        ]:
            if not os.path.exists(f"{self.data_path}/{postfix}"):
                raise FileNotFoundError(f"This file should exist from object creation: {self.data_path}/{postfix}")

        self.data.flush()
        self.row_index.flush()
        self.col_index.flush()

        if output_path is not None:
            raise NotImplementedError("Saving to separate path is not yet implemented.")

        return True

    def number_of_values(self) -> int:
        """Get the total number of values in the array.

        For each index, the length of the corresponding dataframe is counted.

        Returns:
            The sum of lengths of the features in every row
        """
        return sum(self._feature_index.number_of_values())

    def number_of_rows(self) -> int:
        """The number of rows in the dataset.

        Returns:
            The number of rows in the dataset
        Raises:
            ValueError if the length of the number of rows in the feature
            index does not correspond to the number of stored rows.
        """
        if len(self._feature_index) > 0 and self._feature_index.number_of_rows() != self.row_index.size - 1:
            raise ValueError(
                f"""The nuber of rows in the feature index {self._feature_index.number_of_rows()}
                             does not correspond to the number of rows in the row_index {self.row_index.size - 1}"""
            )
        return self._feature_index.number_of_rows()

    def number_nonzero_values(self) -> int:
        """Number of non zero entries in the dataset."""
        return self.data.size

    def __len__(self):
        """Return the number of rows."""
        return self.number_of_rows()

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the row values located and index idx."""
        return torch.from_numpy(np.stack(self.get_row(idx)[0]))

    def number_of_variables(self) -> List[int]:
        """Get the number of features in every entry in the dataset.

        Returns:
            A list containing the lengths of the features in every row
        """
        feats = self._feature_index
        if len(feats) == 0:
            return [0]
        num_vars = feats.column_dims()
        return num_vars

    def shape(self) -> Tuple[int, List[int]]:
        """Get the shape of the dataset.

        This is the number of entries by the the length of the feature index
        corresponding to that variable.

        Returns:
            The number of elements in the dataset
            A list containing the number of variables for each row.
        """
        return self.number_of_rows(), self.number_of_variables()

    def concat(
        self,
        other_dataset: Union[list["SingleCellMemMapDataset"], "SingleCellMemMapDataset"],
    ) -> None:
        """Concatenates another SingleCellMemMapDataset to the existing one.

        The data is stored in the same place as for the original data set. This
        necessitates using _swap_memmap_array.

        Args:
            other_dataset: A SingleCellMemMapDataset or a list of
            SingleCellMemMapDatasets

        Raises:
           ValueError if the other dataset(s) are not of the same version or
           something of another type is passed in.
        """
        # Verify the other dataset or datasets are of the same type.
        match other_dataset:
            case self.__class__():
                other_dataset = [other_dataset]
            case list():
                pass
            case _:
                raise ValueError(
                    f"Expecting either a {SingleCellMemMapDataset} or a list thereof. Actually got: {type(other_dataset)}"
                )

        for dataset in other_dataset:
            if self.version() != dataset.version():
                raise ValueError(
                    f"""Incompatable versions: input version: {dataset.version()},
            this version:  {self.version}"""
                )

        # Set our mode:
        self.mode: Mode = f"{Mode.READ_APPEND}"

        mmaps = []
        mmaps.extend(other_dataset)
        # Calculate the size of our new dataset arrays
        total_num_elements = (self.number_nonzero_values() if self.number_of_rows() > 0 else 0) + sum(
            [m.number_nonzero_values() for m in mmaps]
        )
        total_num_rows = self.number_of_rows() + sum([m.number_of_rows() for m in mmaps])

        # Create new arrays to store the data, colptr, and rowptr.
        with tempfile.TemporaryDirectory(prefix="_tmp", dir=self.data_path) as tmp:
            data_arr, col_arr, row_arr = _create_compressed_sparse_row_memmaps(
                num_elements=total_num_elements,
                num_rows=total_num_rows,
                memmap_dir_path=Path(tmp),
                mode=f"{Mode.CREATE_APPEND}",
                dtypes=self.dtypes,
            )
            # Copy the data from self and other into the new arrays.
            cumulative_elements = 0
            cumulative_rows = 0
            if self.number_of_rows() > 0:
                data_arr[cumulative_elements : cumulative_elements + self.number_nonzero_values()] = self.data.data
                col_arr[cumulative_elements : cumulative_elements + self.number_nonzero_values()] = self.col_index.data
                row_arr[cumulative_rows : cumulative_rows + self.number_of_rows() + 1] = self.row_index.data
                cumulative_elements += self.number_nonzero_values()
                cumulative_rows += self.number_of_rows()
            for mmap in mmaps:
                # Fill the data array for the span of this scmmap
                data_arr[cumulative_elements : cumulative_elements + mmap.number_nonzero_values()] = mmap.data.data
                # fill the col array for the span of this scmmap
                col_arr[cumulative_elements : cumulative_elements + mmap.number_nonzero_values()] = mmap.col_index.data
                # Fill the row array for the span of this scmmap
                row_arr[cumulative_rows : cumulative_rows + mmap.number_of_rows() + 1] = (
                    mmap.row_index + int(cumulative_rows)
                ).data

                self._feature_index.concat(mmap._feature_index)
                # Update counters
                cumulative_elements += mmap.number_nonzero_values()
                cumulative_rows += mmap.number_of_rows()
            # The arrays are swapped to ensure that the data remains stored at self.data_path and
            # not at a temporary filepath.
            _swap_mmap_array(
                data_arr, f"{tmp}/{FileNames.DATA}", self.data, f"{self.data_path}/{FileNames.DATA}", destroy_src=True
            )
            _swap_mmap_array(
                col_arr,
                f"{tmp}/{FileNames.COLPTR}",
                self.col_index,
                f"{self.data_path}/{FileNames.COLPTR}",
                destroy_src=True,
            )
            _swap_mmap_array(
                row_arr,
                f"{tmp}/{FileNames.ROWPTR}",
                self.row_index,
                f"{self.data_path}/{FileNames.ROWPTR}",
                destroy_src=True,
            )
            # Reopen the data, colptr, and rowptr arrays
            self.data = np.memmap(
                f"{self.data_path}/{FileNames.DATA}",
                dtype=self.dtypes[f"{FileNames.DATA}"],
                shape=(cumulative_elements,),
                mode=f"{Mode.READ_APPEND}",
            )
            self.row_index = np.memmap(
                f"{self.data_path}/{FileNames.ROWPTR}",
                dtype=self.dtypes[f"{FileNames.ROWPTR}"],
                shape=(cumulative_rows + 1,),
                mode=f"{Mode.READ_APPEND}",
            )
            self.col_index = np.memmap(
                f"{self.data_path}/{FileNames.COLPTR}",
                dtype=self.dtypes[f"{FileNames.COLPTR}"],
                shape=(cumulative_elements,),
                mode=f"{Mode.READ_APPEND}",
            )

        self.save()