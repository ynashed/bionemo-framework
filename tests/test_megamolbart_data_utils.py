import os

import pytorch_lightning as pl
import torch
from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer

from bionemo.data.molecule.augment import MoleculeEnumeration, MoleculeInputTargetEnumeration
from bionemo.data.molecule.megamolbart_utils import megamolbart_build_train_valid_test_datasets, \
    megamolbart_retro_build_train_valid_test_datasets
from bionemo.model.utils import initialize_distributed_parallel_state
from tests.test_megamolbart_inference import get_cfg

os.environ["PROJECT_MOUNT"] = os.environ.get(
    "PROJECT_MOUNT", '/workspace/bionemo')

PREPEND_DIR = "../examples/tests/"
CONFIG_PATH = os.path.join(PREPEND_DIR, 'conf')
PREPEND_CONFIG_DIR = '../examples/molecule/megamolbart/conf'

TOKENIZER_MODEL = '/model/molecule/megamolbart/vocab/megamolbart.model'
TOKENIZER_VOCAB = '/model/molecule/megamolbart/vocab/megamolbart.vocab'
SEQ_LEN = 512
SEED = 42

TOKENIZER = RegExTokenizer().load_tokenizer(regex_file=TOKENIZER_MODEL, vocab_file=TOKENIZER_VOCAB)
NUM_SAMPLES = {'train': 5, 'val': 5, 'test': 5}


def test_megamolbart_build_train_valid_test_datasets():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='megamolbart_test', config_path=CONFIG_PATH)

    initialize_distributed_parallel_state()
    train_ds, val_ds, test_ds = megamolbart_build_train_valid_test_datasets(cfg=cfg.model.data,
                                                                            train_valid_test_num_samples=NUM_SAMPLES)
    assert len(train_ds) == NUM_SAMPLES['train'] and len(val_ds) == NUM_SAMPLES['val'] \
           and len(test_ds) == NUM_SAMPLES['test']

    assert train_ds[2] == 'Cc1ccccc1CCON'
    assert val_ds[0] == 'C#CCNC(=O)COC[C@@H](N)C#C'
    assert test_ds[4] == 'O[C@H]1COCCN(CCn2cccn2)C1'


def test_megamolbart_retro_build_train_valid_test_datasets():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='megamolbart_downstream_retro_test', config_path=CONFIG_PATH)

    initialize_distributed_parallel_state()
    train_ds, val_ds, test_ds = megamolbart_retro_build_train_valid_test_datasets(cfg=cfg.model.data,
                                                                                  train_valid_test_num_samples=NUM_SAMPLES)
    assert len(train_ds) == NUM_SAMPLES['train'] and len(val_ds) == NUM_SAMPLES['val'] \
           and len(test_ds) == NUM_SAMPLES['test']

    input_name = cfg.model.data.input_name
    target_name = cfg.model.data.target_name
    assert input_name in train_ds[0].keys() and target_name in train_ds[0].keys()

    assert train_ds[3] == {'products': 'Cc1c(C)n(-c2cccnc2)c2ccc(Br)cc12',
                           'reactants': 'Cc1[nH]c2ccc(Br)cc2c1C.Fc1cccnc1'}

    assert val_ds[2] == {'products': 'CC(=O)NCC(=O)N(C)c1ccc(Cl)c(COc2cccc3ccc(C)nc23)c1Cl',
                         'reactants': 'CC(=O)OC(C)=O.Cc1ccc2cccc(OCc3c(Cl)ccc(N(C)C(=O)CN)c3Cl)c2n1'}


def test_molecule_enumeration_collate_fn():
    pl.seed_everything(SEED)
    collate_fn = MoleculeEnumeration(tokenizer=TOKENIZER, seq_length=SEQ_LEN,
                                     pad_size_divisible_by_8=True,
                                     encoder_mask=False, decoder_mask=False,
                                     encoder_augment=False, decoder_augment=False,
                                     canonicalize_input=False).collate_fn

    batch = ['CO[C@H](C[C@@H]1CCC[C@H]1O)c1ccccc1', 'COC[C@H](O)CCNC(=O)CCC[C@@H](C)N',
             'CC[C@](C)(Nc1cnnn1C)C1CC1']

    output = collate_fn(batch)
    expected_keys = ['text_enc', 'enc_mask', 'text_dec', 'dec_mask', 'labels', 'loss_mask', 'target_smiles']
    assert all([k in expected_keys for k in output.keys()])
    assert len(output['target_smiles']) == output['text_enc'].shape[0] and output['text_enc'].shape[0] == len(batch)
    assert output['text_enc'].shape[1] % 8 == 0

    assert not (TOKENIZER.mask_id in output['text_enc'] and TOKENIZER.mask_id in output['text_dec'])
    assert all([output['text_dec'][i][0] == TOKENIZER.bos_id for i in range(len(batch))])
    assert all([target_smi == smi for target_smi, smi in zip(output['target_smiles'], batch)])

    for i in range(len(batch)):
        mask_dec = output['dec_mask'][i]
        mask_dec[0] = 0
        dec_ids = output['text_dec'][i][mask_dec == 1].cpu().detach().numpy().tolist()

        mask_enc = output['enc_mask'][i]
        enc_ids = output['text_enc'][i][mask_enc == 1].cpu().detach().numpy().tolist()

        assert TOKENIZER.ids_to_text([dec_ids])[0] == batch[i] and \
               TOKENIZER.ids_to_text([enc_ids])[0] == batch[i]

    # double-checking masking
    collate_fn = MoleculeEnumeration(tokenizer=TOKENIZER, seq_length=SEQ_LEN,
                                     pad_size_divisible_by_8=True,
                                     encoder_mask=True, decoder_mask=True,
                                     encoder_augment=False, decoder_augment=False,
                                     canonicalize_input=False, mask_prob=0.1, span_lambda=3.0).collate_fn
    output = collate_fn(batch)
    assert (TOKENIZER.mask_id in output['text_enc']) and (TOKENIZER.mask_id in output['text_dec'])

    assert torch.equal(output['text_enc'], torch.tensor(
        [[272, 4, 287, 275, 272, 290, 274, 4, 272, 4, 281, 273, 274, 273,
          273, 273, 4, 273, 274, 0, 0, 0, 0, 0],
         [272, 285, 272, 287, 4, 279, 272, 275, 280, 285, 281, 272, 272, 272,
          290, 4, 4, 0, 0, 0, 0, 0, 0, 0],
         [272, 272, 291, 275, 272, 281, 275, 279, 273, 274, 273, 282, 282, 282,
          274, 272, 281, 272, 274, 272, 272, 274, 0, 0]]))

    assert torch.equal(output['enc_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]))

    assert torch.equal(output['text_dec'], torch.tensor(
        [[2, 4, 4, 287, 275, 272, 290, 274, 272, 272, 272, 287, 274, 285,
          281, 273, 274, 273, 273, 273, 273, 273, 4, 0],
         [2, 4, 272, 287, 275, 4, 272, 279, 272, 4, 281, 272, 272, 272,
          290, 275, 272, 281, 279, 0, 0, 0, 0, 0],
         [2, 272, 272, 291, 275, 272, 281, 275, 279, 273, 274, 273, 282, 4,
          281, 272, 274, 4, 274, 0, 0, 0, 0, 0]]))

    assert torch.equal(output['dec_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]))

    assert torch.equal(output['labels'], torch.tensor(
        [[4, 4, 287, 275, 272, 290, 274, 272, 272, 272, 287, 274, 285, 281,
          273, 274, 273, 273, 273, 273, 273, 4, 3, -1],
         [4, 272, 287, 275, 4, 272, 279, 272, 4, 281, 272, 272, 272, 290,
          275, 272, 281, 279, 3, -1, -1, -1, -1, -1],
         [272, 272, 291, 275, 272, 281, 275, 279, 273, 274, 273, 282, 4, 281,
          272, 274, 4, 274, 3, -1, -1, -1, -1, -1]]))

    assert torch.equal(output['loss_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]))

    assert output['target_smiles'] == ['CO[C@H](C[C@@H]1CCC[C@H]1O)c1ccccc1', 'COC[C@H](O)CCNC(=O)CCC[C@@H](C)N',
                                       'CC[C@](C)(Nc1cnnn1C)C1CC1']


def test_molecule_input_target_enumeration_collate_fn():
    pl.seed_everything(SEED)

    input_name = 'products'
    target_name = 'reactants'
    collate_fn = MoleculeInputTargetEnumeration(tokenizer=TOKENIZER, seq_length=SEQ_LEN,
                                                pad_size_divisible_by_8=True,
                                                encoder_mask=False, decoder_mask=False,
                                                encoder_augment=False, decoder_augment=False,
                                                canonicalize_input=False,
                                                input_name=input_name,
                                                target_name=target_name).collate_fn

    batch = [{'products': 'CCc1cc(N)c(N)cc1Cl', 'reactants': 'CCc1cc([N+](=O)[O-])c(N)cc1Cl'},
             {'products': 'CC(C)(C)c1ccc(CN(CCc2ccc(Cl)c(C(F)(F)F)c2)C(=O)c2cc(Cl)cc3cc[nH]c23)cc1',
              'reactants': 'CC(C)(C)c1ccc(CNCCc2ccc(Cl)c(C(F)(F)F)c2)cc1.O=C(O)c1cc(Cl)cc2cc[nH]c12'}]

    output = collate_fn(batch)
    expected_keys = ['text_enc', 'enc_mask', 'text_dec', 'dec_mask', 'labels', 'loss_mask', 'target_smiles']
    assert all([k in expected_keys for k in output.keys()])
    assert len(output['target_smiles']) == output['text_enc'].shape[0] and output['text_enc'].shape[0] == len(batch)
    assert output['text_enc'].shape[1] % 8 == 0

    assert (TOKENIZER.mask_id not in output['text_enc']) and (TOKENIZER.mask_id not in output['text_dec'])
    assert all([output['text_dec'][i][0] == TOKENIZER.bos_id for i in range(len(batch))])

    assert all([target_smi == smi['reactants'] for target_smi, smi in zip(output['target_smiles'], batch)])

    for i in range(len(batch)):
        mask_dec = output['dec_mask'][i].clone()
        mask_dec[0] = 0
        dec_ids = output['text_dec'][i][mask_dec == 1].cpu().detach().numpy().tolist()

        mask_enc = output['enc_mask'][i]
        enc_ids = output['text_enc'][i][mask_enc == 1].cpu().detach().numpy().tolist()

        assert TOKENIZER.ids_to_text([dec_ids])[0] == batch[i]['reactants'] and \
               TOKENIZER.ids_to_text([enc_ids])[0] == batch[i]['products']

    assert torch.equal(output['text_enc'], torch.tensor(
        [[272, 272, 273, 274, 273, 273, 275, 279, 281, 273, 275, 279, 281, 273,
          273, 274, 304, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [272, 272, 275, 272, 281, 275, 272, 281, 273, 274, 273, 273, 273, 275,
          272, 279, 275, 272, 272, 273, 277, 273, 273, 273, 275, 304, 281, 273,
          275, 272, 275, 303, 281, 275, 303, 281, 303, 281, 273, 277, 281, 272,
          275, 280, 285, 281, 273, 277, 273, 273, 275, 304, 281, 273, 273, 284,
          273, 273, 286, 273, 277, 284, 281, 273, 273, 274, 0, 0, 0, 0,
          0, 0]])
                       )
    assert torch.equal(output['enc_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]))

    assert torch.equal(output['text_dec'], torch.tensor(
        [[2, 272, 272, 273, 274, 273, 273, 275, 309, 275, 280, 285, 281, 305,
          281, 273, 275, 279, 281, 273, 273, 274, 304, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [2, 272, 272, 275, 272, 281, 275, 272, 281, 273, 274, 273, 273, 273,
          275, 272, 279, 272, 272, 273, 277, 273, 273, 273, 275, 304, 281, 273,
          275, 272, 275, 303, 281, 275, 303, 281, 303, 281, 273, 277, 281, 273,
          273, 274, 299, 285, 280, 272, 275, 285, 281, 273, 274, 273, 273, 275,
          304, 281, 273, 273, 277, 273, 273, 286, 273, 274, 277, 0, 0, 0,
          0, 0]]))

    assert torch.equal(output['dec_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]))

    assert torch.equal(output['labels'], torch.tensor(
        [[272, 272, 273, 274, 273, 273, 275, 309, 275, 280, 285, 281, 305, 281,
          273, 275, 279, 281, 273, 273, 274, 304, 3, -1, -1, -1, -1, -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1],
         [272, 272, 275, 272, 281, 275, 272, 281, 273, 274, 273, 273, 273, 275,
          272, 279, 272, 272, 273, 277, 273, 273, 273, 275, 304, 281, 273, 275,
          272, 275, 303, 281, 275, 303, 281, 303, 281, 273, 277, 281, 273, 273,
          274, 299, 285, 280, 272, 275, 285, 281, 273, 274, 273, 273, 275, 304,
          281, 273, 273, 277, 273, 273, 286, 273, 274, 277, 3, -1, -1, -1,
          -1, -1]]))

    assert torch.equal(output['loss_mask'], torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]))

    assert output['target_smiles'] == ['CCc1cc([N+](=O)[O-])c(N)cc1Cl',
                                       'CC(C)(C)c1ccc(CNCCc2ccc(Cl)c(C(F)(F)F)c2)cc1.O=C(O)c1cc(Cl)cc2cc[nH]c12']


def test_megamolbart_retro_collate_fn_masking_config():
    pl.seed_everything(SEED)
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='megamolbart_downstream_retro_test', config_path=CONFIG_PATH)
    collate_fn = MoleculeInputTargetEnumeration(tokenizer=TOKENIZER, seq_length=cfg.model.data.max_seq_length,
                                                pad_size_divisible_by_8=True,
                                                **cfg.model.data).collate_fn
    batch = [{'products': 'CCc1cc(N)c(N)cc1Cl', 'reactants': 'CCc1cc([N+](=O)[O-])c(N)cc1Cl'}]

    output = collate_fn(batch)
    assert (TOKENIZER.mask_id not in output['text_enc']) and (TOKENIZER.mask_id not in output['text_dec'])
