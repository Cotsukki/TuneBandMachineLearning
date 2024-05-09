## Overview
This project involves building a machine learning pipeline to categorize activities based on emotions and other features. The code employs several preprocessing steps, oversampling techniques to address class imbalance, a Random Forest classifier, and extensive evaluation metrics to gauge performance.

### Loader.py Overview

**Introduction**
The `Loader.py` script is designed to streamline the data intake process for our machine learning system. It extracts data from diverse sources, including a Firebase database, and prepares it in a structured format that's ready for subsequent analysis and model training.

**Key Operations**
- **Data Importation**: The script can retrieve data stored in different formats and from various sources, ensuring a versatile and robust data handling capability.
- **Data Normalization**: It standardizes the data to ensure consistency across different datasets, which is crucial for accurate machine learning predictions.

### Detailed Process Description

**Firebase Data Loading**
1. **Connection Setup**:
    - The script begins by establishing a connection to our Firebase database using a JSON configuration file that contains the necessary credentials. This file is crucial as it authenticates our script to access the database securely.

2. **Data Retrieval**:
    - Once connected, the script queries specific data nodes within our Firebase structure. This could include user-generated data, transaction logs, or other relevant datasets depending on the project's scope.
    - Data fetched from Firebase is in JSON format, which is versatile and widely used for web-based data interchange.

3. **Data Transformation**:
    - The retrieved JSON data is then parsed and transformed into a pandas DataFrame. This step is vital as it converts the data into a tabular format, which is easier to manipulate and analyze using Python.

### Preparing Data for Training

**Cleaning and Preprocessing**
- The script applies various data cleaning techniques such as removing incomplete entries, correcting anomalies, and filling missing values. These steps are essential to enhance the quality of the dataset.

**Feature Engineering**
- Additional features are derived from the existing data to provide more insights and improve the predictive quality of the machine learning models.

**Output File Generation**
- After processing, the script generates CSV files that contain the cleaned and structured data. These files are then used as input for the `training_validation.py` script, which handles model training and validation.
- The generation of these files is a critical step as it links the data loading phase with the model training phase, ensuring a smooth workflow.

### Usage and Configuration

**System Requirements**
- Users must ensure that Python is installed on their system along with pandas and other necessary libraries.
- The JSON configuration file for Firebase must be correctly configured and placed in an accessible location.

**Execution**
- To run the script, simply execute it from a Python environment. Ensure that the paths to data sources and the output directory are correctly specified.

**Benefits**
- Automates the tedious tasks of data loading and preprocessing, allowing data scientists to focus on more strategic aspects like model development and analysis.

### training_validation_final.py Overview

## Data Loading and Preprocessing
- **Data Loading**: Data is loaded from a CSV file into a pandas DataFrame.
- **Activity Mapping**: Activities in the dataset are mapped to broader categories to simplify the classification task. A dictionary maps specific activities to their respective categories.
- **Target Encoding**: The categorical target variable (activity categories) is encoded into numerical format using `LabelEncoder` from `sklearn.preprocessing`.

## Feature Engineering
- **One-Hot Encoding**: Categorical features like 'Emotion' are transformed into numerical format through one-hot encoding, which creates binary columns for each category.
- **Feature Combination**: Numerical features and one-hot encoded features are combined to form the complete feature set used for training.

## Handling Class Imbalance
- **SMOTE and Random Oversampling**: Depending on the minimum class size, either SMOTE or Random Oversampling is applied to balance the dataset, helping improve model performance across classes.

## Model Training and Evaluation
- **Pipeline Creation**: A pipeline comprising a standard scaler and a Random Forest classifier is created. The scaler standardizes features to have zero mean and unit variance, while the classifier is used for making predictions.
- **Model Training**: The model is trained using the prepared dataset.
- **Model Evaluation**: The model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, ROC curves for multi-class classification and a confusion matrix are generated to visualize performance.

## Visualizations
- **ROC Curve**: Receiver Operating Characteristic curves for each class are plotted to evaluate the trade-off between true positive rate and false positive rate at various threshold settings.
- **Confusion Matrix**: A confusion matrix is visualized using Plotly to provide a clear view of the model’s performance in terms of correctly and incorrectly classified instances.
- **Learning Curve**: Learning curves are plotted to analyze the model's performance with increasing amounts of training data, which helps in understanding the bias-variance trade-off.

## Functions and Libraries Used
- **Pandas**: For data manipulation and ingestion.
- **NumPy**: For numerical operations on arrays.
- **Matplotlib & Plotly**: For plotting graphs and interactive visualizations.
- **Scikit-learn**: For model building, data preprocessing, and performance evaluation.
- **imblearn**: For applying oversampling techniques to address class imbalance.
- **StratifiedKFold, RandomizedSearchCV**: For robust cross-validation and hyperparameter tuning.

## Setup and Execution
To execute this code, ensure you have Python installed with the required libraries. Adjust the path to the dataset as per your environment setup.

## Conclusion
This documentation covers the methodology, function descriptions, and performance evaluations included in the project. The approach takes into account various aspects of a typical machine learning workflow, from data preprocessing to detailed performance analysis, providing a robust framework for activity classification based on emotion.
