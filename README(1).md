# Heart Disease Classification

A machine learning project that explores and compares **Logistic
Regression** and **K-Nearest Neighbors (KNN)** for binary heart disease
classification using a heart disease dataset.

The project covers exploratory data analysis, data cleaning,
preprocessing, model training, cross-validation, evaluation, and saving
a reusable model pipeline.

> **Disclaimer:** This project is for educational purposes only. It is
> not a clinically validated diagnostic tool and must not be used to
> make medical decisions.

## Project Structure

``` text
Heart-Disease-Classification/
├── heart_disease_classification_reorganized.py
├── heartu.csv
├── heart_disease_knn.pkl
├── requirements.txt
└── README.md
```

-   `heart_disease_classification_reorganized.py` --- main Python
    script.
-   `heartu.csv` --- dataset used by the project.
-   `heart_disease_knn.pkl` --- saved tuned KNN pipeline, including
    preprocessing.
-   `requirements.txt` --- Python dependencies.
-   `README.md` --- project documentation.

## Dataset

The project uses `heartu.csv`, which contains patient-related features
and a binary target variable.

The target is interpreted as:

-   `0` --- negative class
-   `1` --- positive class

The features used by the model are:

  Feature      Description
  ------------ --------------------------------------------------------
  `age`        Age
  `sex`        Sex indicator as encoded in the dataset
  `cp`         Chest pain type
  `trestbps`   Resting blood pressure
  `chol`       Serum cholesterol
  `fbs`        Fasting blood sugar indicator
  `restecg`    Resting electrocardiographic result
  `thalach`    Maximum heart rate achieved
  `exang`      Exercise-induced angina indicator
  `oldpeak`    ST depression induced by exercise relative to rest
  `slope`      Slope of the peak exercise ST segment
  `ca`         Number of major vessels, as represented in the dataset
  `thal`       Thalassemia-related category in the dataset

## Workflow

### 1. Data inspection and cleaning

-   Inspected dataset structure, feature values, target distribution,
    missing values, and duplicates.
-   Removed duplicate rows.
-   Treated `ca = 4` as an unknown-value code and removed those rows.
-   Kept the remaining observations for modeling.

### 2. Exploratory data analysis

EDA included:

-   Target class distribution
-   Numerical feature distributions and boxplots
-   Categorical feature frequency plots
-   Numerical features compared with the target
-   Cross-tabulations for categorical features against the target
-   A correlation heatmap

Some EDA commands are retained as commented-out code in the Python
script so they can be run again when needed.

### 3. Train-test split

The data was split into training and testing sets using:

-   Test size: `20%`
-   Stratification on the target
-   Random state: `42`

The test set was kept separate from KNN hyperparameter selection.

### 4. Preprocessing

A `ColumnTransformer` was used to apply the following transformations:

-   **StandardScaler** for `age`, `trestbps`, `chol`, `thalach`,
    `oldpeak`, and `ca`.
-   **OneHotEncoder** for `cp`, `restecg`, `slope`, and `thal`.
-   **Passthrough** for the binary columns `sex`, `fbs`, and `exang`.

The preprocessing steps were fitted on training data and then applied to
test data to avoid data leakage.

### 5. Models

Two classification approaches were explored:

-   **Logistic Regression** as a baseline classifier.
-   **K-Nearest Neighbors (KNN)** as a baseline and tuned classifier.

KNN was evaluated with different neighbor counts. The project used
stratified 5-fold cross-validation on the training data to compare
candidate values, with recall as the scoring metric. `K = 21` was
selected from the earlier experiment and used for the final tuned KNN
model.

### 6. Evaluation

The models were evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   ROC-AUC

Recall for class `1` was given particular attention because false
negatives represent positive cases that the model predicts as negative.
The metrics are based on the project's held-out test split and should be
interpreted with the small test-set size in mind.

## Results

Results from the project's recorded test-set evaluation:

  -----------------------------------------------------------------------------
  Model            Accuracy      Class 1      Class 1      Class 1      ROC-AUC
                               Precision       Recall     F1-score 
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           80.00%          79%        84.4%         0.82       0.9063
  Regression                                                       

  KNN (K = 5)        75.00%          77%          75%         0.76       0.8516
  baseline                                                         

  Tuned KNN (K       76.67%          ---        87.5%          ---       0.8817
  = 21)                                                            
  -----------------------------------------------------------------------------

For the tuned KNN model, the recorded confusion matrix was:

``` text
[[18, 10],
 [ 4, 28]]
```

This corresponds to 4 false negatives and 28 true positives for class
`1`.

**Project model choice:** Tuned KNN was selected for the project's focus
on class `1` recall. It had higher recall than Logistic Regression on
this test split, while Logistic Regression had higher accuracy and
ROC-AUC. This is a trade-off observed on this particular split, not
proof that KNN will perform better on unseen populations or in clinical
use.

## Saving and using the model

The script saves the complete tuned KNN pipeline to:

``` text
heart_disease_knn.pkl
```

Because the saved object is a pipeline, it includes the preprocessing
steps needed before prediction.

The script also demonstrates loading the saved model and predicting on a
synthetic example.

To load the pipeline in another Python script:

``` python
import joblib

loaded_model = joblib.load("heart_disease_knn.pkl")
prediction = loaded_model.predict(new_patient)
probabilities = loaded_model.predict_proba(new_patient)
```

`new_patient` should be a pandas DataFrame containing the same feature
columns used during training.

Only load pickle/joblib files from sources you trust. Such files can
execute code when deserialized.

## Installation and execution

### Requirements

Use Python and install the dependencies listed in `requirements.txt`.
The project uses:

-   NumPy
-   pandas
-   seaborn
-   Matplotlib
-   scikit-learn
-   joblib

Install dependencies:

``` bash
pip install -r requirements.txt
```

Place `heartu.csv` in the same directory as the Python script, then run:

``` bash
python heart_disease_classification_reorganized.py
```

The script prints the synthetic example prediction and class
probabilities. Some exploratory and evaluation print statements remain
commented out in the script.

## Limitations

-   The evaluation uses a single held-out test split containing 60
    observations, so the reported metrics have sampling uncertainty.
-   The dataset and results do not establish clinical validity.
-   Model-estimated class probabilities are not necessarily calibrated
    probabilities of disease.
-   The chosen model and threshold may not suit every use case.
-   The project does not replace professional medical assessment.

## Future improvements

-   Evaluate performance with repeated cross-validation or additional
    independent data.
-   Examine probability calibration and threshold selection.
-   Add a clean prediction interface for user-supplied records.
-   Improve documentation of dataset provenance and feature encoding.
-   Compare additional models only where there is a clear experimental
    objective.

## Author

**Ashtitva Pandey**\
B.Tech, Computer Science and Engineering (AI & ML)\
VIT Bhopal University
