# Heart Disease Classification Project
# ------------------------------------
# Educational project using Logistic Regression and tuned KNN.
# This model is not a clinically validated diagnostic tool.

# 1. Imports
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# 2. Load Dataset and Initial Inspection
df = pd.read_csv('heartu.csv')

# Optional dataset inspection:
# print(df.shape)
# print(df.info())
# print(df.head())
# print(df.tail())
# print(df.isnull().sum())
# print(df.duplicated().value_counts())
# print(df['target'].value_counts())
# print(df.columns)

# Remove duplicate rows
df.drop_duplicates(inplace=True)


# 3. Feature Groups
num_cols = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak',
    'ca'
]

cat_cols = [
    'sex',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal'
]

One_cols = ['cp', 'restecg', 'slope', 'thal']
binary_cols = ['sex', 'fbs', 'exang']

# Optional value-count inspection:
# for col in cat_cols:
#     print(df[col].value_counts())


# 4. Exploratory Data Analysis (EDA)
# The following plots and tables are optional. Uncomment the analyses
# you want to display when exploring the dataset.

# Target distribution:
# sns.countplot(x=df['target'])
# plt.show()

# Numerical feature distributions:
# for col in num_cols:
#     sns.histplot(df[col], bins=10)
#     plt.title(f'Distribution of {col}')
#     plt.show()
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# Categorical feature distributions:
# for col in cat_cols:
#     sns.countplot(x=df[col])
#     plt.title(f'Distribution of {col}')
#     plt.show()

# Numerical features against target:
# for col in num_cols:
#     sns.boxplot(x=df['target'], y=df[col])
#     plt.title(f'{col} by target')
#     plt.show()

# Categorical features against target:
# for col in cat_cols:
#     print(pd.crosstab(df[col], df['target'], normalize='index') * 100)

# Correlation heatmap:
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()


# 5. Data Cleaning
# In this dataset, ca=4 was treated as an unknown/missing-value code.
ca_4 = df[df['ca'] == 4]
# print(ca_4)

# The project removes rows containing ca=4.
df.drop(df[df['ca'] == 4].index, inplace=True)

# Final checks:
# print(df.shape)
# print(df['target'].value_counts())
# print(df.duplicated().value_counts())
# print(df.isnull().sum())


# 6. Separate Features and Target
X = df.drop(columns=['target'], axis=1)
y = df['target']


# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Inner split used during the earlier manual KNN validation experiment
X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)


# 8. Preprocessing for the Inner Validation Experiment
inner_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('one', OneHotEncoder(handle_unknown='ignore'), One_cols),
    ('binary', 'passthrough', binary_cols)
])

# Fit only on inner training data; transform validation data
X_train_inner_preprocessed = inner_preprocessor.fit_transform(X_train_inner)
X_val_preprocessed = inner_preprocessor.transform(X_val)


# 9. Main Feature Encoding and Scaling
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('one', OneHotEncoder(), One_cols),
    ('binary', 'passthrough', binary_cols)
])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# 10. Logistic Regression Baseline
model = LogisticRegression()
model.fit(X_train_preprocessed, y_train)

y_pred = model.predict(X_test_preprocessed)

# Evaluation:
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(model.predict_proba(X_test_preprocessed)[:10])


# 11. Logistic Regression ROC-AUC and ROC Curve
y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
auc = roc_auc_score(y_test, y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve - Logistic Regression')
# plt.legend()
# plt.show()


# 12. KNN Baseline
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_preprocessed, y_train)

y_pred_knn = model_knn.predict(X_test_preprocessed)

# Evaluation:
# print("Accuracy:", accuracy_score(y_test, y_pred_knn))
# print(confusion_matrix(y_test, y_pred_knn))
# print(classification_report(y_test, y_pred_knn))

K_cols = [1, 3, 5, 7, 9, 11, 15, 21]

# Manual validation experiment from the project:
# for k in K_cols:
#     KNeighborclassifier = KNeighborsClassifier(n_neighbors=k)
#     KNeighborclassifier.fit(X_train_inner_preprocessed, y_train_inner)
#     y_pred_inner = KNeighborclassifier.predict(X_val_preprocessed)
#     print("K =", k, "Accuracy =", accuracy_score(y_val, y_pred_inner))
#     print(confusion_matrix(y_val, y_pred_inner))
#     print(classification_report(y_val, y_pred_inner))


# 13. KNN Baseline ROC-AUC
y_prob_knn = model_knn.predict_proba(X_test_preprocessed)[:, 1]
auc_knn = roc_auc_score(y_test, y_prob_knn)
# print("KNN ROC-AUC:", auc_knn)


# 14. Tuned KNN with Cross-Validation
knn_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('one', OneHotEncoder(handle_unknown='ignore'), One_cols),
        ('binary', 'passthrough', binary_cols)
    ])),
    ('knn', KNeighborsClassifier())
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Cross-validation loop used to compare candidate K values:
# for k in K_cols:
#     knn_pipeline.set_params(knn__n_neighbors=k)
#     scores = cross_val_score(
#         knn_pipeline,
#         X_train,
#         y_train,
#         cv=cv,
#         scoring='recall'
#     )
#     print(scores)
#     print("Mean CV Recall:", scores.mean())

# K=21 was selected from the earlier cross-validation experiment.
knn_pipeline.set_params(knn__n_neighbors=21)
knn_pipeline.fit(X_train, y_train)

y_pred_final = knn_pipeline.predict(X_test)

# Evaluation:
# print("Accuracy:", accuracy_score(y_test, y_pred_final))
# print(confusion_matrix(y_test, y_pred_final))
# print(classification_report(y_test, y_pred_final))


# 15. Compare ROC-AUC of Logistic Regression and Tuned KNN
lr_prob = model.predict_proba(X_test_preprocessed)[:, 1]
lr_auc = roc_auc_score(y_test, lr_prob)

knn_prob = knn_pipeline.predict_proba(X_test)[:, 1]
knn_auc = roc_auc_score(y_test, knn_prob)

# print("Logistic Regression AUC:", lr_auc)
# print("Tuned KNN AUC:", knn_auc)


# 16. Save and Reload the Tuned KNN Pipeline
joblib.dump(knn_pipeline, "heart_disease_knn.pkl")

loaded_model = joblib.load("heart_disease_knn.pkl")
loaded_pred = loaded_model.predict(X_test)

# Verify that reloaded predictions match the original pipeline predictions:
# print(loaded_pred[:10])
# print(y_test.iloc[:10].values)
# print(np.array_equal(loaded_pred, y_pred_final))


# 17. Example Prediction on a Synthetic Input
new_patient = pd.DataFrame([{
    'age': 55,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 240,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 1,
    'ca': 0,
    'thal': 2
}])

prediction = loaded_model.predict(new_patient)
probability = loaded_model.predict_proba(new_patient)

print("Prediction:", prediction[0])
print("Class probabilities:", probability[0])
