# SL-ViT on CIFAR-10 and CIFAR-100

Registered name: `sl_vit`. The implementation follows the author's
[SPT_LSA_ViT repository](https://github.com/aanna0701/SPT_LSA_ViT/tree/54bd796acd28fa11deac21a97b14b154379b34e7)
at revision `54bd796acd28fa11deac21a97b14b154379b34e7`, specifically
`models/create_model.py`, `models/vit.py`, `models/SPT.py`, `main.py`, and the
training utilities. Both SPT and LSA are enabled, corresponding to the author's
`--model vit --is_SPT --is_LSA`.

The model uses 32×32 RGB inputs, 4×4 patches, embedding dimension 192, 9 blocks,
12 heads (16 dimensions per head), and MLP ratio 2. SPT concatenates four
zero-padded diagonal shifts of 2 pixels before patchification, normalization and
projection. LSA masks the attention diagonal and learns one scaling parameter
per head. Classification uses the class token. Dropout is zero, stochastic depth
is a constant 0.1 in every block, and linear layers use Xavier normal
initialization. There is no external pretraining.

## Independent configurations

`configs/sl_vit_cifar10.json` and `configs/sl_vit_cifar100.json` are loaded
automatically by `--model sl_vit --dataset cifar10|cifar100`. They do not inherit
the project's other model or dataset training defaults.

| Setting | CIFAR-10 | CIFAR-100 |
| --- | --- | --- |
| Epochs / batch size | 100 / 128 | 100 / 128 |
| AdamW maximum LR / weight decay | 0.001 / 0.05 | 0.001 / 0.05 |
| AdamW betas / epsilon | (0.9, 0.999) / 1e-8 | (0.9, 0.999) / 1e-8 |
| Linear warmup / cosine minimum LR | 10 epochs / 1e-6 | 10 epochs / 1e-6 |
| Label smoothing | 0.1 | 0.1 |
| Normalization mean | (0.4914, 0.4822, 0.4465) | (0.5070, 0.4865, 0.4409) |
| Normalization std | (0.2470, 0.2435, 0.2616) | (0.2673, 0.2564, 0.2762) |

The author's executable code defaults to LR **0.001**, while section 4.1 of the
[paper](https://arxiv.org/abs/2112.13492) specifies **0.003 for ViT**. These
configurations follow code defaults. To evaluate the paper LR explicitly, add
`--init_lr 0.003`; the actual value is recorded in results and checkpoints.

Training transforms, in order, are horizontal flip, crop with padding 4,
the author's CIFAR10 AutoAugment policy (for **both** datasets), tensor conversion,
normalization, and random erasing. The author's PIL policy semantics are
retained, including its interpolation and fill rules. Erasing probability is
0.25, area is sampled uniformly from 0.02 to 0.4, aspect ratio uniformly from
0.3 to 1/0.3, with 100 attempts and dataset-mean fill **after normalization**.

Every batch has probability 0.5 of mixing. Conditional on mixing, CutMix and
Mixup each have probability 0.5, with Beta(1, 1) weights. CutMix adjusts the loss
weight using the actual clipped box area. Label smoothing applies to both
targets in the mixed loss. AMP and gradient accumulation are disabled, matching
the official FP32 training flow.

The author's repeated augmentation concatenates **three independently shuffled
passes** through the selected subset per epoch. With `drop_last=True`, batch
count is `floor(3 * num_selected / batch_size)`, not `ceil(num_selected / batch_size)`.
For the full 50,000-image dataset, this is 1,171 steps per epoch at batch size 128.
Warmup and cosine are stepped after each optimizer update and use the actual
subset-loader length. The final cycle boundary follows the author's reset to
minimum LR. The sampler uses an epoch-specific local seed to support resumption;
its distribution follows the author, but random ordering is not guaranteed to
match an upstream run bit for bit.

## Commands

Full-data baselines:

```bash
python train_after_selection.py --dataset cifar10 --model sl_vit --mode random --kr 100 --seed 22,42,96
python train_after_selection.py --dataset cifar100 --model sl_vit --mode random --kr 100 --seed 22,42,96
```

Selected-data experiments use the same recipe:

```bash
python train_after_selection.py --dataset cifar10 --model sl_vit --mode learned_group --kr 30,50,70 --seed 22,42,96
python train_after_selection.py --dataset cifar100 --model sl_vit --exp label_noise --mode learned_group --kr 30,50,70 --seed 22,42,96
python train_after_selection.py --dataset cifar100 --model sl_vit --exp corruption --mode learned_group --kr 30,50,70 --seed 22,42,96
```

`noise_train.py --model sl_vit` also delegates to this training implementation,
preserving that entry point's root arguments. Models other than SL-ViT keep that
entry point's existing behavior. SL-ViT supports only CIFAR-10 and CIFAR-100.

Numeric CLI overrides remain available. A short smoke run must explicitly reduce
warmup, e.g. `--epochs 2 --warmup_epochs 0`. Batch sizes or selections that
produce zero complete batches are rejected.

Results include the source revision, optimizer, scheduler, augmentation,
normalization, smoothing, stochastic depth and actual steps per epoch. The
project's **mean of the final 10 epoch accuracies** remains the evaluation
protocol; the author's original script selects the best validation accuracy,
so these summaries are different. Seeds and selection masks follow the project's
experiment protocol. Apply the same SL-ViT recipe to every selection method when
comparing methods.

Checkpoints save the recipe, loader length, selection context and random states,
and reject incompatible recipes on resume. Verification covers forward/backward
passes, attention masking, patch order, scheduler resumption, mixing, and actual
CIFAR-10/100 training loops with synthetic data. Full CIFAR training is a separate
experiment; synthetic checks do not establish reproduction of paper accuracy.
