# Housing Price Prediction Model 🏠

## Overview
This is my **first machine learning model** - a housing price prediction system built with Python and scikit-learn. The model predicts median house values based on various features using the California Housing dataset.

## Project Description

This project implements a complete end-to-end machine learning pipeline for predicting housing prices—a classic regression problem that demonstrates fundamental ML concepts. The system uses the California Housing dataset, which contains 20,640 records with features like location (latitude/longitude), housing age, room counts, population density, and median income to predict median house values.

**The Machine Learning Approach:**
The project employs a Random Forest Regressor, an ensemble learning method that combines multiple decision trees to provide robust and accurate predictions. The model was selected after comparing it with Linear Regression and Decision Tree models using 10-fold cross-validation, demonstrating superior performance in handling complex non-linear relationships in real estate data.

**Data Processing Pipeline:**
A sophisticated preprocessing pipeline handles mixed data types intelligently. Numerical features (rooms, age, population) undergo median imputation to handle missing values, followed by standard scaling to normalize their ranges. Categorical features (ocean proximity) are transformed using one-hot encoding. This stratified approach ensures each feature type is processed optimally while maintaining data integrity throughout the pipeline.

**Key Workflow:**
The system operates in two phases: training and inference. During training, data is split into 80/20 train-test sets using stratified sampling based on income categories to ensure balanced representation. The trained model and preprocessing pipeline are serialized using joblib for efficient deployment. During inference, the same pipeline transforms new data before making predictions, ensuring consistency and reproducibility.

**Significance:**
As my first ML project, this demonstrates understanding of the complete machine learning lifecycle—from data exploration and preprocessing to model training, evaluation, and deployment. It showcases practical implementation of scikit-learn's powerful abstractions and best practices for building production-ready ML systems.

## Features
- 📊 Handles both numerical and categorical features
- 🔧 Complete preprocessing pipeline with imputation and scaling
- 🌲 Random Forest Regressor for accurate predictions
- 💾 Model serialization for deployment
- 📈 Cross-validation support for model evaluation
- ⚡ Efficient batch prediction capability

## Dataset
Uses the California Housing dataset with features such as:
- Latitude & Longitude
- Housing median age
- Total rooms & bedrooms
- Population & households
- Median income
- Ocean proximity

## Installation
1. Clone the repository:
```bash
git clone https://github.com/rohitmane45/housing-price-prediction.git
cd housing-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The model files and CSV data will be generated automatically on first run.

## Usage

### Training the Model
```bash
python main.py
```
The first run will:
- Load the housing dataset
- Split data into training (80%) and test (20%) sets
- Preprocess and scale features
- Train the Random Forest model
- Save the trained model and pipeline

### Making Predictions
On subsequent runs, the script will:
- Load the pre-trained model
- Read input data from `input.csv`
- Generate predictions
- Save results to `output.csv`

## Requirements
- Python 3.7+
- numpy
- pandas
- scikit-learn

## Project Structure
```
.
├── main.py                 # Main training and inference script
├── main_old.py             # Original training script (reference)
├── requirements.txt        # Project dependencies
├── .gitignore              # Files to exclude from Git
├── README.md               # This file
│
├── housing.csv             # Training dataset (generated/local)
├── input.csv               # Test data (generated/local)
├── output.csv              # Predictions (generated/local)
├── model.pkl               # Trained model (generated/local, 137 MB)
└── pipeline.pkl            # Preprocessing pipeline (generated/local)
```

**Note on .gitignore:**
The `.gitignore` file prevents uploading large files to GitHub:
- `*.pkl` files (model.pkl, pipeline.pkl) - Too large (137+ MB)
- `*.csv` files - Can be regenerated from code
- `__pycache__/` - Python cache files
- `venv/` - Virtual environment

## Model Performance
- **Algorithm**: Random Forest Regressor
- **Training/Test Split**: 80/20 stratified split based on income categories
- **Preprocessing**: 
  - Numerical: Median imputation + Standard scaling
  - Categorical: One-Hot encoding

## Learning Outcomes
This was my first ML project where I learned:
- Data preprocessing and feature engineering
- Building sklearn pipelines for reproducible workflows
- Model training and evaluation
- Handling mixed data types (numerical & categorical)
- Model serialization and deployment

## Future Improvements
- Add cross-validation metrics
- Compare with other models (Linear Regression, Gradient Boosting)
- Fine-tune hyperparameters
- Add data visualization
- Create a web API for predictions

## License
This project is open source and available under the MIT License.

## About .gitignore
This project uses a `.gitignore` file to exclude large and auto-generated files from GitHub:
- **Model files** (`*.pkl`) - Too large for GitHub (137+ MB limit)
- **Data files** (`*.csv`) - Can be regenerated when the script runs
- **Python cache** (`__pycache__`, `*.pyc`) - Auto-generated
- **Virtual environment** (`venv/`) - User-specific

This keeps the repository lightweight while maintaining full functionality. When you clone and run `main.py`, all necessary files are generated automatically.

## Author
Created as my first machine learning project to demonstrate end-to-end ML pipeline development.

---

Feel free to use this project as a reference for your own machine learning journey! 🚀
