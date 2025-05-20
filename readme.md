# LingWav2Vec2: Linguistic-augmented wav2vec 2.0 for Vietnamese Mispronunciation Detection

[![Stars](https://img.shields.io/github/stars/tuanio/ling-wav2vec2?style=social)](https://github.com/tuanio/ling-wav2vec2/stargazers)
[![Fork](https://img.shields.io/github/forks/tuanio/ling-wav2vec2?style=social)](https://github.com/tuanio/ling-wav2vec2/network/members)
[![Python 3.7](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.21437%2FInterspeech.2024--1569-blue)](https://doi.org/10.21437/Interspeech.2024-1569)

<picture>
  <img alt="LingWav2Vec2" src="https://firebasestorage.googleapis.com/v0/b/database-7ca5c.appspot.com/o/lingwav2vec2%2FScreenshot%202024-08-29%20at%2010.23.19%E2%80%AFAM.png?alt=media&token=06eb2897-05bc-4c42-956a-478f83e9701b" width="100%">
</picture>

## Overview
LingWav2Vec2 is a novel approach for Vietnamese mispronunciation detection, combining a pre-trained wav2vec 2.0 model with a linguistic encoder. This project achieved top rank in the Vietnamese Mispronunciation Detection (VMD) challenge at VLSP 2023.

## Motivation
- Improve Vietnamese mispronunciation detection and diagnosis (MD&D)
- Address challenges in mispronunciation detection due to limited training data
- Leverage both acoustic and linguistic information for a balanced approach

## Key Features
- Combines wav2vec 2.0 with a linguistic encoder
- Processes raw audio input
- Utilizes canonical phoneme information
- Only 4.3M additional parameters on top of wav2vec 2.0

## Results
- Achieved top-rank on VLSP private test leaderboard
- F1-score of 59.68%, a 9.72% improvement over previous state-of-the-art
- Outperformed more complex models (e.g., TextGateContrast) with fewer parameters
- Balanced use of canonical linguistic information (27.63% relative difference in accuracy)

### 🏆 Competition Results on Private Test

| # | Team Name | F1 | Precision | Recall |
|---|-----------|----|-----------| ------|
| 1 | LossKhongGiam (our) | 57.55 | 55.52 | 59.73 |
| 2 | SpeechHust98 | 55.19 | 41.37 | 82.86 |
| 3 | DaNangNLP | 52.02 | 38.34 | 80.89 |
| 4 | TruongNguyen | 49.27 | 34.51 | 86.07 |
| 5 | TranTuanBinh | 14.90 | 12.88 | 17.68 |

Our team "LossKhongGiam" achieved the highest F1 score and precision metrics, demonstrating the effectiveness of this toolkit in real-world competitive scenarios.# ASR-Toolkit

## Ablation Study
- Non-freezing wav2vec 2.0 CNN layers yielded optimal results
- SpecAugment with specific parameters achieved best F1-score
- Linguistic Encoder significantly boosted performance

## Future Work
- Explore MD&D-specific data augmentation
- Investigate impact of pitch information on Vietnamese mispronunciation detection

## Citation
If you use this work, please cite our paper.

```
@inproceedings{nguyen24b_interspeech,
  title     = {LingWav2Vec2: Linguistic-augmented wav2vec 2.0 for Vietnamese Mispronunciation Detection},
  author    = {Tuan Nguyen and Huy Dat Tran},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2355--2359},
  doi       = {10.21437/Interspeech.2024-1569},
  issn      = {2958-1796},
}
```

## Contact
For questions or collaborations, please contact:
- Tuan Nguyen (Institute for Infocomm Research (I²R), A*STAR, Singapore - nvatuan3@gmail.com)
- Huy Dat Tran (Institute for Infocomm Research (I²R), A*STAR, Singapore).

## Acknowledgements
This work will be poster presented at INTERSPEECH 2024.
