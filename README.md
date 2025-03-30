# ISM

By [Jeffrey Ouyang-Zhang](https://jozhang97.github.io/), [Chengyue Gong](https://sites.google.com/view/chengyue-gong), [Yue Zhao](https://zhaoyue-zephyrus.github.io), [Philipp Krähenbühl](http://www.philkr.net/), [Adam Klivans](https://www.cs.utexas.edu/users/klivans/), [Daniel J. Diaz](http://danny305.github.io)

This repository is an official implementation of the paper [Distilling Structural Representations into Protein Sequence Models](https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2).

**TL; DR.** ESM2 with enriched structural representations

## Download ISM

The model URL contains ISM accessible in both huggingface and ESM2 format.  These models can be used as a drop-in replacements for ESM2 (see [Quickstart](#quickstart)).
Most users should use the first model (ISM-650M-UC30PDB). The second model (ISM-650M-UC30) is for users who do not want a model trained on PDB (e.g. for benchmarking).

| Name | Layers | #params | Dataset | Model URL |
|---------------------------|:--------:|:---------:|:------------------------------------:|:------------------------------:|
| ISM-650M-UC30PDB | 33 | 650M | Uniclust30 + PDB | https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb |
| ISM-650M-UC30 | 33 | 650M | Uniclust30 | https://huggingface.co/jozhang97/ism_t33_650M_uc30 |
| ISM-3B-UC30 | 36 | 3B | Uniclust30 | https://huggingface.co/jozhang97/ism_t36_3B_uc30 |
| ISM-C-300M | 30 | 300M | Uniclust30 + PDB | https://huggingface.co/jozhang97/ismc-300m-2024-12 |
| ISM-C-600M | 36 | 600M | Uniclust30 + PDB | https://huggingface.co/jozhang97/ismc-600m-2024-12 |

## Quickstart

This quickstart assumes that the user is already working with ESM2 and is interested in replacing ESM with ISM. First, download ISM.
```bash
# recommended
huggingface-cli download jozhang97/ism_t33_650M_uc30pdb --local-dir /path/to/save/ism

# alternative
git clone https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb
```

If the user is starting from [fair-esm](https://github.com/facebookresearch/esm), add the following lines of code.
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
ckpt = torch.load('/path/to/ism_t33_650M_uc30pdb/checkpoint.pth')
model.load_state_dict(ckpt)
```

If the user is starting from [huggingface](https://huggingface.co/facebook/esm2_t33_650M_UR50D), replace the model and tokenizer with the following line of code.
```python
from transformers import AutoTokenizer, AutoModel
config_path = "/path/to/ism_t33_650M_uc30pdb/"
model = AutoModel.from_pretrained(config_path)
tokenizer = AutoTokenizer.from_pretrained(config_path)
```

Please change `/path/to/ism_t33_650M_uc30pdb` to the path where the model is downloaded.


## Installation

The following reproduction setup walks through how to structure-tune, fine-tune on a downstream task and evaluate our model.
Prepare your conda environment.
```bash
conda create -n ism python=3.10 -y
pip install -r requirements.txt
```

## ISM Structure-tuning

ISM is initialized from ESM2 and fine-tuned on structural tokens. Download the dataset from [here](https://huggingface.co/datasets/jozhang97/structure-tuning-uc30pdb) (131 GB uncompressed, 22GB compressed).

If you are using slurm, use the following command. This trains in roughly 1 day.
```bash
cd plm_train
python submitit_train.py --nodes 32 --ngpus 1 --dist_eval \
    --loss_func allmergedce_ce \
    --data_path /path/to/dataset \
    --job_dir logs/%j_ism
```

If you are training on an 8 GPU machine, the following training script corresponds to the above command.
```bash
cd plm_train
torchrun --nproc_per_node=8 main_train.py --accum_iter 4 --dist_eval \
    --loss_func allmergedce_ce \
    --data_path /path/to/dataset
    --output_dir logs/ism
```



## Structural Benchmark Evaluation
Here, we show how to reproduce our performance on the secondary structure and binding residues datasets. The datasets are make available at `plm_eval/data`.

To retrain the models, use the following commands.
```bash
cd plm_eval
torchrun --nproc_per_node=8 main_train.py \
    --data_path data/binding_residue/development_set/train.csv \
    --eval_data_path data/binding_residue/development_set/test.csv \
    --freeze_at 33 --lr 1e-4 \
    --finetune_backbone /path/to/ism_t33_650M_uc30/checkpoint.pth \
    --output_dir logs/ism_binding_residue

torchrun --nproc_per_node=4 main_train.py \
    --data_path data/secondary_structure/train.csv \
    --eval_data_path data/secondary_structure/test.csv \
    --freeze_at 33 \
    --finetune_backbone /path/to/ism_t33_650M_uc30/checkpoint.pth \
    --output_dir logs/ism_secondary_structure
```

To evaluate our models, add `--resume /path/to/ism_finetuned --eval` to the above commands. The models fine-tuned for secondary structure is available [here](https://utexas.box.com/s/cetl0rr22on6c0yejxkglqtp0r6lzakc) and for binding residues [here](https://utexas.box.com/s/8btt411wy2l57ebakif8dgj8shcmpfbk). The models are 2.4GB each.
(Note that here we evaluate ISM structure-tuned on AlphaFold structures in Uniclust30 only to avoid data leakage.)


## License

This project builds off [ESM](https://github.com/facebookresearch/esm/). Please refer to their original licenses for more details.


## Citing ISM
If you find ISM useful in your research, please consider citing:

```bibtex
@article{ouyangzhang2024distilling,
  title={Distilling Structural Representations into Protein Sequence Models},
  author={Ouyang-Zhang, Jeffrey and Gong, Chengyue and Zhao, Yue and Kr{\"a}henb{\"u}hl, Philipp and Klivans, Adam and Diaz, Daniel J},
  journal={bioRxiv},
  doi={10.1101/2024.11.08.622579},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
