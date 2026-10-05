# SHL Hiring Assessment 2026

Audio-based machine learning solution for the SHL Hiring Assessment 2026.

## Project Overview

This project predicts assessment-related labels from audio recordings using machine learning.

## Dataset

* Training samples: 769 audio files
* Test samples: 216 audio files
* Audio sampling rate: 16 kHz
* Target: Regression label

## Feature Extraction

The audio files were converted into numerical features using MFCCs.

* 13 MFCC features
* Mean of each MFCC
* Standard deviation of each MFCC
* Total: 26 features per audio file

## Machine Learning Approach

The following models were evaluated:

1. Random Forest Regressor
2. Extra Trees Regressor

An 80/20 train-validation split was used with `random_state=42`.

### Validation Results

| Model         |    MAE |   RMSE |
| ------------- | -----: | -----: |
| Random Forest | 0.6055 | 0.7477 |
| Extra Trees   | 0.5646 | 0.7037 |

Extra Trees performed better on the validation set and was used for the final predictions.

## Final Model

* Model: Extra Trees Regressor
* Number of estimators: 300
* Random state: 42
* Parallel processing: `n_jobs=-1`

## Submission

The final submission contains predictions for 216 test audio files.

File:

`submission.csv`

## Repository Contents

* `submission.csv` — final prediction file
* `README.md` — project documentation
* `solution.ipynb` — solution notebook/code

## Kaggle Result

Public Kaggle Score: **0.7509**

## Limitations

This solution uses MFCC-based audio features with tree-based machine learning models. It does not use a heavy deep-learning audio pipeline.
