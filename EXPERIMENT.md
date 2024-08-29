fine-w2v2base-bs8-ep300-lr2e-5-non_freeze-lr_cosine-non_freeze-spec_aug
- follow TIMIT in wav2vec2.0 paper for 40k steps update (400 epoch)
    mask_time_prob=0.065,
    mask_time_length=10,
    mask_feature_prob=0.012,
    mask_feature_length=64

# Experiment steps
1. Finetune freeze CNN on 95% data
2. Finetune non-freeze on 95% data
3. Finetune non-freeze on 100% data with spec augment (TIMIT config in wav2vec2.0)
4. Let's see :))) maybe edit wav2vec2 loss by calculating CTC Loss on tonal and text also 

# 

default: no tonal, non_freeze, lr cosine

Things to test:
- The amount of SpecAug
    - base from 0.075_10_0.004_64 (Best PER: 0.0908 - 9600)
    - reduce to 0.05_10_0.004_20 (Best PER: {} - {})
    - increase to 0.075_10_0.012_64 
    => When watching the loss, increase SpecAug policy make the model works better
- Batch size:
    - 8 (get from best above)
    - 32
- LayerNorm / RMSNorm
    - 
- LR Cosine / LR Linear
- Focal loss 
    alpha: 0.99 | 0.75 | 0.5 | 0.25 (3)
    gamma: 0.5 | 1 | 2 ( 3 | 5)
    + to test for focal loss:
        + spec aug policy: 0.05_10_0.004_40
        + speed perturb
        + epoch 100
        + batch size 16
        + lr 2e-5
        + warmup 0.1, cosine decay
    => best: alpha = 0.99, gamma = 2
