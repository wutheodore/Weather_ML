# Weather_ML — Machine Learning Experiments

## Project Overview

Weather_ML is a Python-based machine learning project focused on weather data classification and model evaluation.

The project explores different machine learning algorithms and feature engineering techniques to compare model performance.

## Dataset

The experiments use the following weather features:

- Temperature
- Humidity
- Wind Speed
- Cloud Cover
- Pressure

The target variable is `Rain`.

The dataset file `weather_forecast_data.csv` is required to run the experiments. It is not currently included in this repository.

## Models and Experiments

- Random Forest Classification
- Support Vector Machine (SVM) with an RBF kernel
- K-Nearest Neighbors (KNN)
- Feature Engineering
- Model Evaluation and Diagnostic Analysis
- Baseline Model Comparison

## Evaluation Metrics

The experiments use several evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

Some experiments use stratified 5-fold cross-validation to evaluate model performance.

## Technologies

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## How to Run

1. Install a compatible Python version.
2. Install the required packages:

   `pip install numpy pandas scikit-learn matplotlib`

3. Place `weather_forecast_data.csv` in the project directory.
4. Run an experiment, for example:

   `python experiment_4_random_forest.py`

Check each script for its specific requirements before running it.

## Project Status

This is an academic and personal learning project involving machine learning experimentation and model evaluation.

## Author

Wutheodore

Computer Science Student | Python | Machine Learning | Data Analysis
