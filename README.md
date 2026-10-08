# SEAL: Similarity-guided End-to-end Augmentation and Labeling

SEAL is a course research project for semi-supervised few-shot image
classification. It extends the relational embedding pipeline in
[RENet](https://github.com/dahyun-kang/renet) with an end-to-end support-set
augmentation procedure: unlabeled examples receive pseudo-labels from
support-to-unlabeled similarity, high-confidence examples are selected, and
their confidence scores are used when forming augmented class prototypes.

The implementation focuses on the standard 5-way miniImageNet benchmark.

## Method

Each episode contains a labeled support set `S`, an unlabeled set `U`, and a
query set `Q`.

1. A ResNet-12 backbone and Self-Correlational Representation (SCR) module
   extract spatial features.
2. Cross-Correlational Attention (CCA) measures the relation between every
   support and unlabeled example.
3. Each unlabeled example receives a pseudo-label from its most similar class.
   The top `m` examples for every class are added to the support set.
4. Original support examples receive weight 1; pseudo-labeled examples receive
   their CCA confidence as weight. The weighted augmented prototypes are then
   used to classify the query set.
5. Query classification loss trains the complete pipeline, allowing the task
   objective to refine the features used for pseudo-labeling.

The central implementation is in:

- `models/support_aug.py`: pseudo-label selection and confidence weighting;
- `models/renet.py`: SCR/CCA relation modeling and weighted support pooling;
- `models/dataloader/pair_dataset.py`: paired labeled/unlabeled episodes;
- `train.py` and `test.py`: end-to-end training and episodic evaluation.

## Reported results

Mean accuracy with a 95% confidence interval on miniImageNet:

| Method | 5-way 1-shot | 5-way 5-shot |
| --- | ---: | ---: |
| RENet baseline | 66.950 +/- 0.438 | 82.325 +/- 0.307 |
| FSL-PRS | 75.55 +/- 0.90 | 85.16 +/- 0.51 |
| LST | 70.1 +/- 1.9 | 78.7 +/- 0.8 |
| **SEAL (ours)** | **75.741 +/- 0.520** | **86.069 +/- 0.286** |

The ablation in the course report compared unweighted mean aggregation with
confidence-weighted aggregation:

| Prototype aggregation | 5-way 1-shot | 5-way 5-shot |
| --- | ---: | ---: |
| Mean | 75.063 +/- 0.536 | 85.793 +/- 0.280 |
| **Confidence-weighted** | **75.741 +/- 0.520** | **86.069 +/- 0.286** |

These are the results recorded in the project report. Reproduction may vary
with hardware, random seeds, dataset preparation, and checkpoint choice.

## Environment

The original experiments used Linux, CUDA 11.0, PyTorch 1.7.1, and a
ResNet-12 backbone. Create the provided Conda environment with:

```bash
conda env create -f environment.yml
conda activate renet_iccv21
```

The environment is intentionally kept close to the original RENet setup and
contains older pinned dependencies. A CUDA-capable GPU is required by the
current training and evaluation code.

## Dataset layout

Download miniImageNet using the helper script:

```bash
cd datasets
bash download_miniimagenet.sh
cd ..
```

Arrange the data as follows:

```text
datasets/miniimagenet/
├── images/
└── split/
    ├── train.csv
    ├── train_unlabeled.csv
    ├── val.csv
    ├── val_unlabeled.csv
    ├── test.csv
    └── test_unlabeled.csv
```

The repository includes the split CSV files but not the image data.

## Training

The report setup uses 5-way episodes, 15 query examples per class, 15
unlabeled examples per class, and the top 5 pseudo-labeled examples per class.

For 1-shot training:

```bash
python train.py \
  -dataset miniimagenet \
  -gpu 0 \
  -shot 1 \
  -query 15 \
  -unlabeled 15 \
  -pseudo_topk 5 \
  -extra_dir seal-5w1s \
  -temperature_attn 5.0 \
  -lamb 0.25 \
  -no_wandb
```

For 5-shot training:

```bash
python train.py \
  -dataset miniimagenet \
  -gpu 0 \
  -shot 5 \
  -query 15 \
  -unlabeled 15 \
  -pseudo_topk 5 \
  -extra_dir seal-5w5s \
  -temperature_attn 5.0 \
  -lamb 0.25 \
  -milestones 40 50 \
  -max_epoch 60 \
  -no_wandb
```

Convenience scripts for these configurations are available under
`scripts/train/`.

## Evaluation

Evaluation loads `max_acc.pth` from
`checkpoints/miniimagenet/<shot>shot-5way/<run-name>/`.

```bash
# 5-way 1-shot
python test.py -dataset miniimagenet -gpu 0 -shot 1 \
  -unlabeled 15 -pseudo_topk 5 -extra_dir seal-5w1s \
  -temperature_attn 5.0

# 5-way 5-shot
python test.py -dataset miniimagenet -gpu 0 -shot 5 \
  -unlabeled 15 -pseudo_topk 5 -extra_dir seal-5w5s \
  -temperature_attn 5.0
```

By default, testing averages 2,000 randomly sampled episodes.

## Repository structure

```text
.
├── common/                 # metrics, arguments, and run utilities
├── datasets/               # download scripts and split definitions
├── models/
│   ├── dataloader/         # episodic labeled/unlabeled sampling
│   ├── support_aug.py      # SEAL support-set augmentation
│   ├── renet.py            # weighted relational embedding model
│   ├── scr.py              # self-correlation module
│   └── cca.py              # cross-correlation attention module
├── scripts/                # training and evaluation commands
├── train.py
├── test.py
└── environment.yml
```

## Scope and limitations

- The reported SEAL experiments were conducted on miniImageNet. Other dataset
  loaders and scripts are inherited from RENet and were not part of the final
  SEAL evaluation.
- Pseudo-label confidence is correlated with correctness but does not guarantee
  it; incorrect high-confidence samples remain a source of noise.
- The code targets the original CUDA/PyTorch environment and has not been
  modernized for current PyTorch releases.

## Attribution

This repository is a derivative research project built on the official
[RENet implementation](https://github.com/dahyun-kang/renet) by Dahyun Kang,
Heeseung Kwon, Juhong Min, and Minsu Cho. The original MIT license and copyright
notice are preserved in `LICENSE`. The SEAL-specific additions are the paired
semi-supervised episode pipeline, pseudo-label-based support augmentation, and
confidence-weighted prototype computation.

If you use the underlying RENet implementation, please cite:

```bibtex
@inproceedings{kang2021renet,
  author    = {Kang, Dahyun and Kwon, Heeseung and Min, Juhong and Cho, Minsu},
  title     = {Relational Embedding for Few-Shot Classification},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2021}
}
```

## License

This code is distributed under the MIT License inherited from RENet. See
`LICENSE` for details.
