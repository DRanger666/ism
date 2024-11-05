from pathlib import Path
from Bio import SeqIO

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def protein_collate_fn(batch, alphabet, args):
    ## convert sequences into tokens https://github.com/facebookresearch/esm/blob/0b59d87ebef95948c735b1f7aad463dc6dfa991b/esm/data.py#L253
    # <SOS> SEQ <EOS> <PAD>
    inputs = {}
    alphabet = alphabet[1]
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([(str(i), item['wt_seq']) for i, item in enumerate(batch)])

    # mask tokens
    drop_mask = torch.empty_like(batch_tokens).bernoulli_(args.mask_ratio).bool()
    drop_mask[batch_tokens == alphabet.padding_idx] = 0
    drop_mask[batch_tokens == alphabet.eos_idx] = 0
    drop_mask[batch_tokens == alphabet.cls_idx] = 0
    while (~drop_mask).all():
        print(f'did not drop with {batch_tokens.shape=}, resampling')
        drop_mask = torch.empty_like(batch_tokens).bernoulli_(args.mask_ratio).bool()
        drop_mask[batch_tokens == alphabet.padding_idx] = 0
        drop_mask[batch_tokens == alphabet.eos_idx] = 0
        drop_mask[batch_tokens == alphabet.cls_idx] = 0

    tokens_w_masked = batch_tokens * (~drop_mask) + alphabet.mask_idx * drop_mask
    inputs['tokens'] = tokens_w_masked
    inputs['labels'] = batch_tokens
    inputs['drop_mask'] = drop_mask

    for k in ['aae_tokens', 'evo_tokens']:
        if k not in batch[0]:
            continue
        inputs[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=0)

    for k in ['fp', 'wt_seq']:
        if k not in batch[0]:
            continue
        inputs[k] = [item[k] for item in batch]

    return inputs

# NB: not used, but simple enough and could be useful for developers
class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, args):
        fasta_seqs = SeqIO.parse(open(data_path),'fasta')
        self.seqs = [str(fasta.seq) for fasta in fasta_seqs]
        self.crop_len = args.crop_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        L = len(seq)

        # random crop
        if L > self.crop_len and self.crop_len > 0:
            start_idx = np.random.randint(0, L-self.crop_len)
            seq = seq[start_idx:start_idx+self.crop_len]
            L = len(seq)

        return {
            'wt_seq': seq,
        }


class SequenceStructureTokensDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, args):
        data_path = Path(data_path)
        self.data = list(data_path.rglob('shard*/*.pth')) + list(data_path.rglob('shard*/*.pt'))
        print(f'Loaded {data_path} with {len(self.data)} samples')

        self.chain_min_len = 32
        self.crop_len = args.crop_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp = self.data[idx]
        info = torch.load(fp, weights_only=False)

        meta = info['meta']
        evo_tokens = info['evo_tokens'] + 1  # 0 is padding
        aae_tokens = info['aae_tokens'] + 1  # 0 is padding

        # check if chain with enough AA exists
        chain_ids = meta[:,2]
        chain_id_list = [chain for chain, counts in zip(*np.unique(chain_ids, return_counts=True)) if counts > self.chain_min_len]
        if len(chain_id_list) == 0:
            print(f'{idx=} has no chain with >{self.chain_min_len} AA')
            rand_item = np.random.choice(len(self))
            return self.__getitem__(rand_item)

        # sample a chain
        chain_id = np.random.choice(chain_id_list)
        mask = chain_ids == chain_id
        meta = meta[mask]
        evo_tokens = evo_tokens[mask]
        aae_tokens = aae_tokens[mask]

        # reorder by residue number
        evo_tokens = evo_tokens[meta[:,3].argsort()]
        aae_tokens = aae_tokens[meta[:,3].argsort()]
        meta = meta[meta[:,3].argsort()]
        seq = ''.join(map(d.get, meta[:,4]))
        L = len(seq)

        # random crop
        if L > self.crop_len and self.crop_len > 0:
            start_idx = np.random.randint(0, L-self.crop_len)
            seq = seq[start_idx:start_idx+self.crop_len]
            evo_tokens = evo_tokens[start_idx:start_idx+self.crop_len]
            aae_tokens = aae_tokens[start_idx:start_idx+self.crop_len]
            L = len(seq)

        ret = {
            'wt_seq': seq,
            'evo_tokens': evo_tokens,
            'aae_tokens': aae_tokens,
        }
        return ret


def create_dataset(args, train=True) -> torch.utils.data.Dataset:
    fp = args.data_path if train else args.eval_data_path
    if 'sequence_structure_tokens' in args.ds_type:
        ds = SequenceStructureTokensDataset(fp, args)
    elif args.ds_type == 'fasta':
        ds = FastaDataset(fp, args)
    else:
        raise ValueError(f'Unknown dataset type: {args.ds_type}')
    return ds
