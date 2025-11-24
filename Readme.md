
## Project Title

H.O.P.E.  (Huntington’s Outcome Prediction Engine)

## Overview

This project applies Machine Learning techniques to a clinical dataset of Huntington's Disease patients. By analyzing genetic markers (CAG repeats), neurological data (Brain Volume Loss), and clinical symptoms, the system builds predictive models to estimate disease staging, functional capacity, and symptom severity. This tool demonstrates how data science can support neurodegenerative disease research.

## Features

Multi-Model Analysis: Includes Classification, Regression, and Clustering algorithms.

Stage Prediction: Uses Random Forest to predict disease progression (Early/Middle/Late).

Functional Capacity Estimation: Uses Linear Regression to quantify patient independence.

Unsupervised Discovery: Uses K-Means clustering to find natural patient groupings without labeled stages.

Data Visualization: (Optional extension) Correlation analysis between genetic mutations and physical symptoms.


## Technologies & Tools Used

Language: Python 3.x

Libraries:

1) pandas 

2) scikit-learn 

3) numpy 

4) seaborn / matplotlib 

Environment: Jupyter Notebook / VS Code / Google Colab


## Steps to Install & Run

Prerequisites: Ensure Python is installed.

Install Requirements:

pip install pandas, numpy, scikit-learn, matplotlib seaborn.


 Data: Ensure Huntington_Disease_Dataset.csv is in the same directory as the script.



## Run the Script:

	python huntington_analysis.py


Output: The script will output accuracy scores, regression metrics (MSE, R2), and classification reports to the console.



## Testing Instructions

Unit Testing: Verify that the dataset loads correctly without encoding errors.

Model Validation: Check the printed Classification Report for Model 1. Precision and Recall should be reasonable (> 0.60).

Sanity Check: Ensure that the Linear Regression for Functional Capacity produces values within the 0-100 range.