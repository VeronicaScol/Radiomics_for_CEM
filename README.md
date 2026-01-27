# Keras Machine Learning for contrast-enhanced mammography 
Keras implementation of RetinaNet object detection as described in [Combining deep learning and handcrafted radiomics for classification of suspicious lesions on contrast-enhanced mammograms](link)
by Manon P.L. Beuque, Marc B.I. Lobbes, Yvonka van Wijk, Yousif Widaatalla, Sergey Primakov, Michael Majer, Corinne Balleyguier, Henry C. Woodruff, Philippe Lambin.

The repository consists in two parts: 
- The first part allows the user to train/test a Mask-RCNN model which predicts the location of a candidate suspicious lesion and predicts a label (benign/malignant).
- The second part allows the user to train/test a machine learning model using radiomics on either the ground truth suspicious lesions or the candidate lesion predicted in the first part. It returns a predicted label benign or malignant.

## Installation
This repository is using Python 3.7 and works with keras version 2.2.4.
Install the packages in requirements.txt file to start using the repository.

## Delineations and predictions with Mask-RCNN
### Training and annotations format
The format of the annotations used for training is described in [Keras MaskRCNN](https://github.com/fizyr/keras-maskrcnn) repository.
Provide 0 for 'benign' and 1 for 'malignant'.
The original dataset used was provided as .mha but DICOM images could also be used after small modification of the preprocessing functions.
The data needs to be preprocessed with preprocessing.utils.preprocessing before usage.
The model can be trained with train_delineation_model.py. 

### Predictions
The delineation predictions are obtained with inference_delineation_and_dl_predictions.py.
Use a csv file to indicate the location of the low energy and corresponding recombined images.
The file should contain "low_energy_paths" and "recombined_paths" headers.


## Predictions using radiomic features
To train and test a machine learning model using radiomic features, the features needs to be extracted from the data.
You will need a csv file containing the following headers: "path_mask","path_low_energy", "path_recombined", "outcome" (which needs to be 0 or 1) to generate a csv file containing the features to use.

## Disclaimer and license
The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
The tool is provided for internal non-commercial academic research purposes only. See license for more information.

[link]: https://github.com/fizyr/keras-maskrcnn

---

# Appendix - Project group MAI3004

---

## Getting started

Before training any models, the provided datasets need to be concatenated and the external dataset needs to be refactored to reflect a binary ground truth.

The first step is to run the file called 01_merge_LE_RE_features.ipynb using the following files:
- train_features_true_mask.csv
- train_features_true_mask_low_energy.csv
- test_features_true_mask.csv
- test_features_true_mask_low_energy.csv
- external_features_true_mask.csv
- external_features_true_mask_low_energy_v2.csv
- annotations_external_dataset.csv

The output of this will be the following files:
- train_merged_LE_RE.csv
- test_merged_LE_RE.csv
- external_merged_LE_RE.csv (intermediate file- further processed in external ground truth section)
- external_test.csv

Moving forward, the training dataset refers to train_merged_LE_RE.csv and the testing dataset is either test_merged_LE_RE.csv for initial testing and external_test.csv for external validation of the results.

## Exploratory data analysis: optional

### eda_radiomics

**Files needed:**

- train_merged_LE_RE.csv,
- test_merged_LE_RE.csv,
- external_test.csv 
This file gives an overview of the radiomics data. The workflow includes the following:
1) Dataset size + class balance (+ bar chart)
    - Duplicate row checks (exact + near-duplicates)
2) Missing values per feature (full listing + CSV export + histogram)
3) Variance per feature (+ histogram + low-variance count)
4) Feature scale heterogeneity (means/stds before normalization) with symlog plots
5) After Z-score normalization: check means~0, stds~1 (+ histograms)
6) Correlation redundancy: histogram of abs(Spearman rho) + % pairs > 0.85
7) Evidence of some signal: ANOVA F-score distribution (optional single-feature AUC)
8) Low-dimensional projection: 3D PCA scatter colored by class + PCA loadings inspection

At the moment, the file is coded to only consider "original" radiomics features (no log_sigma or wavelet features). 

### eda_clinical_data

**Files needed:**
- combined_clinical_features_train_processed.csv
- combined_clinical_features_test_processed.csv
- combined_clinical_features_external_processed.csv

This file includes the following exploratory elements: 
1) Structure of the dataset
2) Missing data
3) Outliers using IQR +/- 1.5xIQR method and boxplot visualisation
4) Demographic data: Age; Pregnancies; No.of children; Family History; Personal History; Menopause status; Cup size
5) Diagnostic Overview
6) Clinical risk factors vs. Diagnosis
7) Correlation Analysis
8) Medication analysis
9) Summary
    
## Radiomics machine learning Pipeline (File structure & usage)

This repository contains a modular machine learning pipeline for training radiomics (& radiomics + clinical features) based classifiers. The pipeline is organised into utility modules and notebooks for experiment runs and outputs. This ensures reproducibility, consistency and fair model comparion.

### Key files for radiomics ML workflow

#### 1. radiomics_pipeline/utils.py
**Purpose:** core preprocessing and evaluation utilities
Contains reusable functions:
- Feature preprocessing
- Probability-based predictors
- optimal threshold selection
- metric computation with bootstrap confidence intervals

#### 2. feature_selection_utils.py
**Purpose:** implement different feature selection strategies
Contains
- Filter methods (ANOVA)
- Wrapper methods (RFE and RFECV)
- Embedded methods (tree based and L1-regularised logistic regression)

Each selector returns a standardised output so that they can be switched out and applied to different models.

#### 3. modeling_pipeline_utils.py
**Purpose:** experiment orchestration and model comparison
This file defines the full modeling pipeline:
- Loads training and testing feature CSVs
- Applies consistent preprocessing
- Runs multiple combinations of:
  - feature selection methods
  - machine learning models (e.g. XGBoost, Random Forest, SVM, Logistic Regression)
- Trains models using selected features
- Determines optimal decision thresholds on training data
- Evaluates performance on test data with:
    - ROC-AUC and other metrics (with confidence intervals)
    - Confusion matrices (raw and normalized)
- Aggregates results for side-by-side comparison
     This file acts as the single entry point for running radiomics ML experiments
   

#### 4. Final_modelling_output.ipynb
>Ensure the data concatenation has been done before running.

**Output:**
- Metrics_table: summary of performance metrics for all model-selector combinations
- results: per-experiment outputs
- jaccard: feature overlap matrix to assess feature selection stability
