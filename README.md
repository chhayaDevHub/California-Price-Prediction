
# California House Price Prediction

## Project Overview
A Machine Learning project that predicts California house prices using Linear Regression and Random Forest.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Machine Learning Models
- Linear Regression
- Random Forest Regressor
- Decision Tree
## Features
- Predicts house prices using input features.
- Provides predictions through a REST API.
- Uses saved machine-learning models for inference.

## Run Locally

### 1. Create a virtual environment
python -m venv venv

### 2. Activate it on Windows
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train and save the models
python train.py

### 5. Start the API
uvicorn main:app --reload

### 6. Open the API documentation
http://127.0.0.1:8000/docs

## Example Use Case
The API accepts housing features and returns a predicted house price.

## Author
Machine Learning Project
