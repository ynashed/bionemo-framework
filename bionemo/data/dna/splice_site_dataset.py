from enum import IntEnum
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Tuple, Any, Callable
from bionemo.data.dataloader.kmer_collate import KmerBertCollate
from bionemo.data.fasta_dataset import ConcatFastaDataset, FastaDataset
from bionemo.data.validation_dataset import DataFrameTransformDataset
from bionemo.core import BioNeMoDataModule
import pandas as pd
from functools import partial
from bionemo.data.utils import expand_dataset_paths
from bionemo.data.utils import NeMoUpsampling
import os
import math


class SpliceSite(IntEnum):
    DONOR = 0
    ACCEPTOR = 1
    NEGATIVE = 2


class InstanceDataset(Dataset):
    def __init__(self, list_: List, kind: SpliceSite):
        """Represents a single splice site

        Args:
            list_ (List): Contains a List of a single kind of splice site
            kind (SpliceSite): The kind of splice site the dataset represents
        """
        self.list_ = list_
        self.kind = kind

    def __getitem__(self, idx):
        return {'coord': self.list_[idx], 'kind': int(self.kind)}

    def __len__(self):
        return len(self.list_)


class TranscriptDataset(Dataset):
    def __init__(self, site_tuple: Tuple[List[int], List[int], List[int]],
                 transcript: str):
        """Dataset that represents a single transcript's donor and acceptors,
        and negative examples.

        Args:
            site_tuple (Tuple[List[int], List[int], List[int]]): Contains a
                list of Donors, Acceptors, and Negative sites, respectively.
            site (str): Name of the splice site
        """
        self.transcript = transcript
        ds_0 = InstanceDataset(site_tuple[0], SpliceSite.DONOR)
        ds_1 = InstanceDataset(site_tuple[1], SpliceSite.ACCEPTOR)
        ds_2 = InstanceDataset(site_tuple[2], SpliceSite.NEGATIVE)
        self._dataset = ConcatDataset([ds_0, ds_1, ds_2])

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item.update({'transcript': self.transcript})
        return item

    def __len__(self):
        return len(self._dataset)


class ChrSpliceSitesDataset(Dataset):
    def __init__(self, chr_sites: Dict[str, Tuple[List[int], List[int], List[int]]], id_: str):
        """Represents all of the splice sites present in a chromosome

        Args:
            chr_sites (Dict[str, Tuple[List[int], List[int], List[int]]]):
                Keys are transcript names and the entries are tuples of splice
                    sites of Donor, Acceptor, and Negative, respectively
            id_ (str): Chromosome id/name
        """
        self.chr_sites = chr_sites
        self._dataset = ConcatDataset(
            TranscriptDataset(sites, name) for name, sites in chr_sites.items()
        )
        self.id_ = id_

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item.update({'id': self.id_})
        return item

    def __len__(self):
        return len(self._dataset)

def get_start_end(coord, length):
    """Gets start and end coordinates of a subsequence of `length` centered around `coord`

    Args:
        coord (int): Center of window
        length (int): Width of window

    Returns:
        Tuple[int, int]: Start and end coordinates of window, respectively
    """
    start = int(coord - math.ceil(length / 2)) + 1
    end = int(coord + math.floor(length / 2)) + 1
    return start, end

def delistify_single_arg(fn: Callable[[List[Any]], Dict]) -> Callable[[Any], Dict]:
    """Makes a function that runs on a list able to be run on a single entry
    per dict key without being called as a list

    Args:
        fn (Callable[[List[Any]], Dict]): Function that maps a list

    Returns:
        Callable[[Any], Dict]: Function that operates on an entry of that list
    """
    def wrapper_fn(arg):
        arg = [arg]
        ret_val = fn(arg)
        return {key: value[0] for key, value in ret_val.items()}
    return wrapper_fn

def fetch_bert_dna(row: pd.Series, dataset: FastaDataset, bert_prep, length: int):
    """Fetches and preprocesses data based

    Args:
        row (pd.Series): Row with `coord` and `id` attributes
        dataset (FastaDataset): Fasta to get DNA text from
        bert_prep (Callable[[str], Any]): Preprocesses the DNA from the FASTA
        length (int): Length of DNA to fetch from `dataset`

    Returns:
        Dict[str, Any]: Preprocessed representation of the fetched DNA
    """
    mid = row.coord
    start, end = get_start_end(mid, length)
    text = dataset.fetch(row.id, start, end)
    return bert_prep(text)

def get_target(row: pd.Series):
    """sets the target from a dataframe row using the .kind attribute

    Args:
        row (pd.Series): row to set target from

    Returns:
        Dict: Dict containing 'target' as the key and the value gotten as the value
    """
    return {'target': row.kind}

def get_autosomes(root_directory, pattern):
    """Generates filenames for autosomes based on a pattern using {}

    Args:
        root_directory (str): Filepath for root directory
        pattern (str): str containing '{}' where '{}' will be replaced with the
            autosome id

    Returns:
        List[str]: List of filepaths to autosomes
    """
    paths = expand_dataset_paths(
        '(' + pattern.format('[1..9]') + ',' + pattern.format('[10..22]') + ')', None)
    return [os.path.join(root_directory, path) for path in paths]


class SpliceSiteDataModule(BioNeMoDataModule):

    def __init__(self, cfg, trainer, model):
        self.cfg = cfg
        self.model = model
        self.train_file = cfg.data.train_file
        self.val_file = cfg.data.get('val_file')
        self.test_file = cfg.data.get('test_file')
        fasta_files = self.get_fasta_files()
        self.length = cfg.seq_length
        self.fasta_dataset = ConcatFastaDataset(
            fasta_files, self.length, backend='memory',
        )
        super().__init__(cfg, trainer)
        self.init_num_samples()

    def get_fasta_files(self):
        fasta_directory = self.cfg.data.fasta_directory
        pattern = self.cfg.data.fasta_pattern
        return get_autosomes(fasta_directory, pattern)

    def train_dataset(self):
        gff_dataset = self._create_dataset(self.train_file)
        return gff_dataset

    def val_dataset(self):
        if self.val_file is None:
            return []
        gff_dataset = self._create_dataset(self.val_file)
        return gff_dataset

    def test_dataset(self):
        if self.test_file is None:
            return []
        gff_dataset = self._create_dataset(self.test_file)
        return gff_dataset

    def _create_dataset(self, filename):
        bert_prep = delistify_single_arg(KmerBertCollate(
            self.model.tokenizer,
            modify_percent=0,
            seq_length=self.length,
            pad_size_divisible_by_8=True,
        ).collate_fn)

        gff_dataset = DataFrameTransformDataset(
            filename,
            functions = [
                partial(
                    fetch_bert_dna,
                    dataset=self.fasta_dataset,
                    bert_prep=bert_prep,
                    length=self.length,
                ),
                get_target,
            ],
            read_csv_args={'dtype': {'id': str}}
        )

        return gff_dataset

    def sample_train_dataset(self, dataset):
        """Creates an upsampled version of the DNABERT training dataset

        """
        num_samples = self.train_num_samples
        dataset_dir = self.cfg.dataset_path
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.train_file,
            index_mapping_dir=dataset_dir,
            name='train',
            )

        return dataset

    def sample_val_dataset(self, dataset):
        if len(dataset) == 0:
            return dataset
        num_samples = self.val_num_samples
        dataset_dir = self.cfg.dataset_path
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.val_file,
            index_mapping_dir=dataset_dir,
            name='val',
            )
        return dataset