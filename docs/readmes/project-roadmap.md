# Project Roadmap

## Stage 1: Understand the Data

The first goal is not to train a complex model. It is to understand how the tracking rows, contact labels, players, steps, plays, and videos are connected.

Main checks:

- dataset dimensions;
- missing values;
- positive/negative class balance;
- unique games and plays;
- contact types;
- player-ground versus player-player events;
- tracking coverage around labelled contacts.

## Stage 2: Build Tracking Features

The first modelling table should be interpretable.

Initial features:

- player-pair distance;
- player speed;
- relative speed;
- acceleration;
- relative acceleration;
- direction/orientation differences;
- temporal context around the event.

The exact feature list will be expanded only after the raw dataset has been inspected.

## Stage 3: PyTorch Tracking Baseline

Start with a small multilayer perceptron.

Why:

1. it gives a reproducible baseline;
2. it is easy to debug;
3. it shows whether engineered tracking features contain useful signal;
4. it provides a fair comparison before adding video.

## Stage 4: Video

The video branch will be added after the tracking pipeline is stable.

Possible inputs include:

- player crops;
- helmet detections;
- short temporal clips around a contact;
- video-derived spatial features.

The exact architecture will depend on data quality and compute requirements.

## Stage 5: Multimodal Fusion

Compare three conditions under the same grouped split:

1. Tracking only
2. Video only
3. Tracking + video

The multimodal model should only be considered better if it improves held-out performance under the same evaluation protocol.

## Stage 6: Evaluation

Because contacts are imbalanced, accuracy is not enough.

Primary metrics:

- sensitivity / recall;
- specificity;
- precision / PPV;
- F1;
- AUROC;
- AUPRC;
- MCC;
- false-positive and false-negative counts.

## Validation Principle

The split should be grouped by `game_play`, not by random rows, so that rows from the same play do not appear in both training and evaluation sets.
