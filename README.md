# Financial Default Prediction: Credit Scoring Model
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)

## 📋 Project Overview
This project was developed as part of the **M2 Scoring Course**.It implements a complete credit scoring pipeline to predict corporate financial difficulty (`yd`) based on a dataset of 181 firms. This analysis is based on the work of Edward I. Altman *"Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"* in 1968. The analysis follows a rigorous econometric approach, comparing **Linear Probability (LPM)**, **Logit**, and **Probit** models.

## 👨‍💻 Replication
Part of this project involves using an **LLM** to replicate the original code I wrote in other languages and to correct it. You will find an **R** file that does the same thing than the code I wrote in Python. To replicate the original paper from Altman, I have to answer 24 questions about the transformation and analysis of data. I am required to use the even rows as training data and the odd rows as validation data.

## 💾 Python Ressources
Pandas: Used for loading the defaut2000.csv dataset and performing critical data cleaning.

NumPy: Employed for advanced numerical operations, including handling infinite values, generating ranges for probability density functions, and creating masks for correlation heatmaps.

Statsmodels: This is the core engine for the econometric analysis; it is used to fit and compare the Linear Probability Model (LPM), Logit, and Probit models.

SciPy (stats): Used to conduct the mandatory statistical tests for the project: Jarque-Bera for normality, Levene for equality of variances, and T-tests for mean differences.

Scikit-Learn (sklearn): Utilized for data preprocessing (StandardScaler) and for calculating the Area Under the ROC Curve (AUC) to evaluate model performance on both training (even rows) and validation (odd rows) samples.

Matplotlib.pyplot: The primary tool for generating all required visual reports, including the distribution plots for each explanatory variable and the final ROC curve comparisons.

Seaborn: Integrated to create high-level statistical visualizations, such as the PairGrid for ratio distributions and the annotated correlation heatmap with T-statistics

##
As a student if you have any remarks or comments about this work contact me ! 😁
