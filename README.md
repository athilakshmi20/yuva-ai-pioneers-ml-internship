# Student Performance Analysis and Machine Learning

## Project Overview
This repository contains projects completed as part of the YuvaIntern AI Pioneers Machine Learning Internship. The UCI Student Performance dataset is used to practice data preprocessing, supervised learning, model evaluation, and visualization.

## Dataset
- **Source:** UCI Machine Learning Repository
- **File:** `student-mat.csv`
- **Records:** 395 students
- **Features:** 33 columns

## Week 1: Python for Machine Learning and Data Preprocessing
**File:** `week1_preprocessing.py`

Activities:
- Loaded and explored the dataset using Pandas.
- Checked missing values and duplicate records.
- Encoded categorical features.
- Normalized numerical features.
- Generated visualizations of final grades and study time.

Outputs:
- `student_performance_preprocessed.csv`
- `final_grade_distribution.png`
- `average_grade_by_studytime.png`

## Week 2: Supervised Machine Learning Models
**File:** `week2_supervised_ml.py`

### Regression
Trained and compared Linear Regression and Random Forest Regressor models.

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 1.647 | 2.378 |
| Random Forest Regressor | 1.201 | 2.015 |

Random Forest Regressor achieved lower prediction errors on the test split.

### Classification
Trained and compared Logistic Regression and Decision Tree models to classify students as pass or fail. A final grade of 10 or above out of 20 was considered a pass.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.873 | 0.957 | 0.849 | 0.900 |
| Decision Tree | 0.886 | 0.978 | 0.849 | 0.909 |

The Decision Tree achieved higher accuracy and F1-score on the test split.

### Week 2 Output Files
- `week2_regression_comparison.csv`
- `week2_classification_comparison.csv`
- `week2_regression_comparison.png`
- `week2_classification_comparison.png`

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## How to Run
1. Install the required libraries:
   `python -m pip install pandas numpy matplotlib scikit-learn`
2. Place the dataset and Python scripts in the same folder.
3. Run Week 1:
   `python week1_preprocessing.py`
4. Run Week 2:
   `python week2_supervised_ml.py`

## Note
The model scores represent the current train/test split and may vary with different data splits. These projects are intended for learning and experimentation.
