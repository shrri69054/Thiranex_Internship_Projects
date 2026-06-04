📊 Data Science Task Portfolio — Shrrivathsan
Panimalar Engineering College
A collection of four end-to-end data science projects covering data cleaning, machine learning classification, exploratory data analysis, and health analytics with predictive modelling.
---
📁 Projects Overview
Task	Project	Domain	Key Technique
TASK 1	Data Cleaning & Visualization	Education	Data Cleaning, Feature Engineering
TASK 2	Predictive Modeling (ML)	Education	Binary Classification (LR, DT, RF)
TASK 3	EDA — Titanic Dataset	Historical / Social	Exploratory Data Analysis
TASK 4	Health Data Project — Diabetes	Healthcare	EDA + ML (LR, RF, GBM, SVM)
---
TASK 1: Data Cleaning & Visualization
File: `data_cleaning_project.py`
Dataset: Synthetic Student Performance Dataset (300 students, 5 departments)
Due: 13 May 2026
What It Does
An end-to-end data pipeline that generates a realistic dirty dataset, applies systematic cleaning, engineers new features, and produces a 9-panel dashboard.
Data Cleaning Steps
Removes 12 injected duplicate rows
Standardizes department labels (e.g., `cse` → `CSE`) and drops invalid entries
Normalizes internship values (`yes`, `YES`, `no` → `Yes` / `No`)
Flags and removes age outliers (valid range: 17–30)
Removes CGPA outliers (valid range: 0–10)
Imputes missing values using median (numeric) and mode (categorical)
Feature Engineering
Feature	Description
`avg_score`	Mean of Math, Science, and English scores
`grade`	Letter grade binned from `avg_score` (F / D / C / B / A)
Dashboard Charts (dashboard.png)
Grade Distribution — Donut chart
Average CGPA by Department — Bar chart
Score Distributions — KDE overlay (Math, Science, English)
Attendance vs CGPA — Scatter with trend line
Internship vs Average Score — Box plot
Correlation Heatmap — 6 numeric features
Students per Department — Horizontal bar chart
Grade Distribution by Department — Stacked percentage bar
Cleaning Summary — Stats panel
Output Files
File	Description
`dashboard.png`	9-panel student performance dashboard
`cleaned_data.csv`	Cleaned and feature-engineered dataset
Setup & Run
```bash
pip install pandas numpy matplotlib seaborn
python data_cleaning_project.py
```
---
TASK 2: Predictive Modeling Using Machine Learning
File: `ml_project.py`
Dataset: Synthetic Student Placement Dataset (800 students, 7 features)
Due: 20 May 2026
What It Does
A full ML pipeline for binary classification — predicting whether a student gets placed — using three algorithms with cross-validation, ROC analysis, and feature importance.
Dataset Features
Feature	Description
`cgpa`	Cumulative GPA (5.0–10.0)
`iq`	IQ score (85–145)
`internship`	Has internship (0/1)
`projects`	Number of projects (0–5)
`backlogs`	Number of backlogs (0–4)
`comm_skill`	Communication skill score (1–5)
`department`	CSE / ECE / MECH / CIVIL / IT
Models Trained
Model	Preprocessing
Logistic Regression	StandardScaler applied
Decision Tree	Raw features (max_depth=6)
Random Forest	Raw features (150 estimators, max_depth=8)
Evaluation Metrics
Accuracy, Precision, Recall, F1-Score
5-Fold Cross-Validation Accuracy
ROC Curve & AUC
Confusion Matrix
Dashboard Charts (ml_dashboard.png)
Model Performance Comparison — All 4 metrics grouped bar
Confusion Matrices — One per model
ROC Curves — All models overlaid
5-Fold CV Accuracy ± Std — Bar with error bars
Feature Importance — Random Forest
CGPA Distribution — Placed vs Not Placed histogram
Model Summary Panel — Best model stats
Setup & Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python ml_project.py
```
---
TASK 3: Exploratory Data Analysis — Titanic
File: `eda_titanic.py`
Dataset: Synthetic Titanic Dataset (891 passengers, realistic distributions)
What It Does
A comprehensive EDA of the Titanic dataset across three multi-panel figures covering survival patterns, demographic breakdowns, correlations, and deep-dive analysis.
Dataset Features
`PassengerId`, `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Cabin`, `Ticket`
Engineered Features
Feature	Description
`FamilySize`	SibSp + Parch + 1
`IsAlone`	1 if FamilySize == 1, else 0
`AgeBin`	Age grouped: Child / Teen / Adult / Middle / Senior
`FareBin`	Fare quartile: Low / Mid / High / Premium
Output Charts
`eda_titanic_overview.png` — 8-panel dashboard:
Survival split donut
Survival rate by passenger class
Survival rate by sex
Passenger count by class & sex
Age distribution by survival outcome
Fare distribution (log scale) by outcome
Family size vs survival rate (dual axis)
Survival rate by port of embarkation
`eda_titanic_correlations.png` — 2-panel correlation view:
Feature correlation matrix (lower triangle heatmap)
Survival % by Age Group × Passenger Class (pivot heatmap)
`eda_titanic_deepdive.png` — 6-panel deep dive:
Age vs Fare scatter coloured by survival
Fare distribution by class (box plot)
Survival rate by age group
Outcomes by class & sex (stacked bar)
Missing values percentage chart
Key insights summary panel
Key Findings
Women had a significantly higher survival rate than men
1st class passengers survived at a higher rate than 3rd class
Children (under 12) had elevated survival probability
Passengers from Cherbourg had higher survival rates than Southampton
Solo travellers had lower survival rates than those travelling with family
Setup & Run
```bash
pip install pandas numpy matplotlib seaborn
python eda_titanic.py
```
---
TASK 4: Real-World Health Data Project — Diabetes
File: `health_eda_model.py`
Dataset: Pima Indians Diabetes (synthetic, medically calibrated, 768 patients)
What It Does
A complete healthcare data science pipeline: realistic dataset generation with clinically accurate distributions, missing data handling, feature engineering, multi-model ML training, and a 4-figure reporting suite.
Dataset Features
Feature	Clinical Significance
`Pregnancies`	Number of pregnancies (gestational diabetes risk)
`Glucose`	Plasma glucose concentration (strongest predictor)
`BloodPressure`	Diastolic blood pressure (mm Hg)
`SkinThickness`	Triceps skinfold thickness (mm)
`Insulin`	2-hour serum insulin (mu U/ml)
`BMI`	Body Mass Index
`DiabetesPedigree`	Family history / genetic risk function
`Age`	Age in years
`Outcome`	Target: 1 = Diabetic, 0 = Non-Diabetic
Missing Data Handling
Biological zeros are treated as missing values and imputed using class-conditional median (separate medians for diabetic and non-diabetic groups):
Column	Missing Rate
Insulin	~49%
SkinThickness	~30%
BloodPressure	~4.5%
BMI	~1.4%
Glucose	~0.7%
Engineered Features
Feature	Formula
`GlucoseBMI`	Glucose × BMI
`AgePregRatio`	Age / (Pregnancies + 1)
`InsulinRes`	Glucose / (Insulin + 1)
Models Trained
Model	Notes
Logistic Regression	L2, max_iter=1000, scaled input
Random Forest	200 estimators, max_depth=6
Gradient Boosting	200 estimators, lr=0.05, max_depth=4
SVM (RBF kernel)	Probability calibrated, scaled input
Evaluation
5-Fold Stratified Cross-Validation (AUC)
Test set: Accuracy, Precision, Recall, F1, AUC
ROC Curves + Precision-Recall Curves
Permutation importance for models without `feature_importances_`
Output Charts
File	Contents
`health_01_eda.png`	Feature distributions by diabetes status + class split donut
`health_02_correlations.png`	Correlation heatmap, violin plots, Glucose vs BMI scatter
`health_03_models.png`	AUC comparison, metrics bar, ROC/PR curves, confusion matrix, feature importance, CV scores
`health_04_conclusions.png`	Full conclusions report: dataset overview, EDA insights, model results, top risk factors, clinical recommendations
Key Findings
Glucose is the strongest individual predictor of diabetes
BMI ≥ 30 + Glucose ≥ 140 represents the highest risk zone
All 4 models exceeded the 0.80 AUC clinical utility threshold
High recall is prioritised over precision for medical screening use cases
Setup & Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python health_eda_model.py
```
---
🛠 Global Prerequisites
Python 3.8+
pip
Install all dependencies at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```
---
👤 Author
Shrrivathsan
Panimalar Engineering College
