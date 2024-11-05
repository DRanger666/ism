from pathlib import Path

import numpy as np
import pandas as pd

import torch

UNKNOWN_VALUE = 999
d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
one_to_three = {v: k for k, v in d.items()}
three_letters = np.array(sorted(list(d.keys())))
three_letters_list = list(three_letters)
aa = sorted(d.values())
one_letters = np.array(aa)
one_letters_list = list(aa)

class SeqDetDatatset(torch.utils.data.Dataset):
    def __init__(self, fp, args, train=True):
        self.name = Path(fp).stem
        self.args = args
        self.df = pd.read_csv(fp)

        cols_to_eval = ['res_ids', 'label', 'mask']
        for col in cols_to_eval:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(eval)
        self.train = train
        print(f'Loaded {len(self.df)} uniprot entries from {fp}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        return row

def protein_collate_fn(batch, alphabet, args):
    ## convert sequences into tokens https://github.com/facebookresearch/esm/blob/0b59d87ebef95948c735b1f7aad463dc6dfa991b/esm/data.py#L253
    # <SOS> SEQ <EOS> <PAD>
    inputs = {}
    alphabet = alphabet[1]

    batch_converter = alphabet.get_batch_converter()
    if 'esm' in args.backbone:
        _, _, batch_tokens = batch_converter([(i, pdb.sequence) for i, pdb in enumerate(batch)])

    if 'secondary' in args.data_path:
        ss_label = torch.zeros_like(batch_tokens[:,1:-1], dtype=torch.long)
        valid_ss_mask = torch.zeros_like(batch_tokens[:,1:-1], dtype=torch.bool)
        for i, pdb in enumerate(batch):
            ss_label[i, :len(pdb.label)] = torch.tensor(pdb.label)
            valid_ss_mask[i, :len(pdb.label)] = torch.tensor(pdb['mask'])
        label_dict = {
            'pdb_id': [pdb.pdb_id for pdb in batch],
            'ss_label': ss_label,
            'valid_ss_mask': valid_ss_mask,
        }
    else:
        bind_label = torch.zeros_like(batch_tokens[:,1:-1], dtype=torch.float32)
        for i, pdb in enumerate(batch):
            bind_label[i, pdb.res_ids] = 1.
        label_dict = {
            'uniprot': [pdb.uniprot for pdb in batch],
            'bind_label': bind_label,
        }

    pad_mask = batch_tokens[:,1:-1] <= 2

    return {
        'tokens': batch_tokens,
        'pad_mask': pad_mask,
        'seqs': [pdb.sequence for pdb in batch],
        **label_dict,
        **inputs,
    }
