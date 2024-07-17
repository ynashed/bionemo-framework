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
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from .label2id_tokenizer import Label2IDTokenizer


__all__ = ["GeneTokenizer"]


class GeneTokenizer(Label2IDTokenizer):
    """Initializes the GeneTokenizer object."""

    cls_token: str = "[CLS]"
    mask_token: str = "[MASK]"
    pad_token: str = "[PAD]"
    sep_token: str = "[SEP]"
    ukw_token: str = "[UKW]"
    special_tokens: Tuple[str, str, str, str, str] = (cls_token, mask_token, pad_token, sep_token, ukw_token)

    def __init__(self, vocab: Dict[str, int], gene_to_ens: Dict[str, str]):
        # Sets up vocab/decode_vocab dictionaries, parent class is sateful.
        super().__init__()
        assert set(self.special_tokens).issubset(
            set(vocab.keys())
        ), f"Vocab must contain all of {self.special_tokens}, missing {set(self.special_tokens) - set(vocab.keys())}"
        self.gene_to_ens = deepcopy(gene_to_ens)
        self.ens_to_gene = {v: k for k, v in self.gene_to_ens.items()}
        self.vocab = deepcopy(vocab)
        self.decode_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_medians_and_genes_dicts(cls, median_dict: Dict[str, float], gene_to_ens: Dict[str, str]):
        '''Creates a tokenizer from a median dictionary'''
        tokens = list(cls.special_tokens) + list(median_dict.keys())
        vocab = cls._build_vocab(tokens)
        return cls(vocab, gene_to_ens)

    @staticmethod
    def _build_vocab(strings: Union[List[str], str]):
        '''We override the parent because complete strings are tokens. Otherwise has the same behavior.'''
        vocab = {}
        if isinstance(strings, str):
            strings = [strings]

        for token in strings:
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding ID.

        Args:
            token (str): The token to be converted.

        Returns:
            int: The ID corresponding to the token.
        """
        return self.vocab.get(token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def class_id(self) -> int:
        return self.token_to_id(self.cls_token)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return super().tokens_to_ids(tokens)

    def save_vocab(self, vocab_file):
        '''Saves the vocabulary as a newline delimieted vocabulary file, each line represents an int -> token mapping. line number is assumed to be the integer.'''
        vocab_dir = os.path.dirname(vocab_file)
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir, exist_ok=True)  # ensure the dir exists but be ok with race conditions.

        to_serialize = {}
        to_serialize['vocab'] = self.vocab
        to_serialize['gene_to_ens'] = self.gene_to_ens

        with open(vocab_file, 'w') as f:
            json.dump(to_serialize, f)

    @classmethod
    def from_vocab_file(cls, vocab_file):
        '''This method adds a layer on the constructor in the case we are working from a filename instead of a dictionary'''
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file {vocab_file} not found, run preprocessing to create it.")

        with open(vocab_file) as f:
            to_deserialize = json.load(f)
            vocab = to_deserialize['vocab']
            gene_to_ens = to_deserialize['gene_to_ens']

        tokenizer = GeneTokenizer(vocab, gene_to_ens)
        return tokenizer

    def gene_tok_to_ens(self, gene: str) -> str:
        """
        Converts a gene token to its corresponding Ensembl ID.

        Args:
            gene (str): The gene token to be converted.

        Returns:
            str: The Ensembl ID corresponding to the gene token.
        """
        return self.gene_to_ens[gene]

    def ens_tok_to_gene(self, ens: str) -> str:
        """
        Converts an Ensembl token to a gene name.

        Args:
            ens (str): The Ensembl token to be converted.

        Returns:
            str: The corresponding gene name.
        """
        return self.ens_to_gene[ens]

    def genes_to_enss(self, genes: List[str]) -> List[str]:
        """Converts a list of gene names to Ensembl IDs.

        Args:
            genes (List[str]): A list of gene names.

        Returns:
            List[str]: A list of corresponding Ensembl IDs.

        Raises:
            ValueError: If a gene name is not found in the gene_to_ens dictionary.
        """
        ens_ids = []
        for gene in genes:
            if gene in self.gene_to_ens:
                ens_ids.append(self.gene_to_ens[gene])
            else:
                raise ValueError(f"{gene} not found")
        return ens_ids

    def enss_to_genes(self, ensemble_ids: List[str]) -> List[str]:
        """Converts a list of ensemble IDs to gene names.

        Args:
            ensemble_ids (List[str]): A list of ensemble IDs.

        Returns:
            List[str]: A list of gene names corresponding to the ensemble IDs.

        Raises:
            ValueError: If an ensemble ID is not found in the mapping.
        """
        genes = []
        for ens_id in ensemble_ids:
            if ens_id in self.ens_to_gene:
                genes.append(self.ens_to_gene[ens_id])
            else:
                raise ValueError(f"{ens_id} not found")
        return genes