# Linear Regression with TensorFlow

## Overview
This project demonstrates the implementation of a linear regression model using TensorFlow. It utilizes the popular machine learning library to predict outcomes based on input data. The project is particularly focused on using categorical and numerical features for prediction.

## Key Features
- **Data Handling**: Uses pandas to read and process data from CSV files.
- **Feature Columns**: Creation of categorical and numeric feature columns using TensorFlow.
- **Model Training and Evaluation**: Training a linear regression model using TensorFlow's estimator API and evaluating its performance.
- **Prediction**: Making predictions based on the trained model.

## Dataset
The project uses two datasets:
- `train.csv`: Used for training the linear regression model.
- `eval.csv`: Used for evaluating the model's performance.

## Implementation
- **TensorFlow Estimators**: The project uses TensorFlow's high-level API, `tf.estimator`, to create and train the linear regression model.
- **Input Function**: A custom input function is defined to convert the pandas DataFrame into a TensorFlow Dataset object, which is then used for training and evaluation.
- **Feature Engineering**: The script demonstrates how to handle categorical and numerical data by converting them into TensorFlow feature columns.

## Usage
1. Ensure you have TensorFlow, NumPy, pandas, and Matplotlib installed.
2. Run the `main.py` script to train and evaluate the model.
3. The script will output the accuracy of the model and a sample prediction.

## Technologies Used
- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib
