# LingWav2Vec2: Linguistic-augmented wav2vec 2.0 for Vietnamese Mispronunciation Detection

<picture>
  <img alt="LingWav2Vec2" src="https://github.com/yourusername/yourrepository/raw/main/images/poster_light.png" width="100%">
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

## Ablation Study
- Non-freezing wav2vec 2.0 CNN layers yielded optimal results
- SpecAugment with specific parameters achieved best F1-score
- Linguistic Encoder significantly boosted performance

## Future Work
- Explore MD&D-specific data augmentation
- Investigate impact of pitch information on Vietnamese mispronunciation detection

## Citation
If you use this work, please cite our paper.

## Contact
For questions or collaborations, please contact:
- Tuan Nguyen (Institute for Infocomm Research (I²R), A*STAR, Singapore - nvatuan3@gmail.com)
- Huy Dat Tran (Institute for Infocomm Research (I²R), A*STAR, Singapore).

## Acknowledgements
This work will be poster presented at INTERSPEECH 2024.
