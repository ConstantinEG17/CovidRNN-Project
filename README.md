# CovidRNN-Project

This project is a deep learning-based time-series analysis focused on predicting COVID-19 deaths using LSTM (Long Short-Term Memory) neural networks. We extend the project by incorporating univariate and multivariate time-series analysis techniques and feature selection methods from COVID-19 datasets.

## Project Overview

The project explores several machine learning and deep learning techniques, including:

1. **Feature Selection using Lasso Regression**: Determines the most relevant factors influencing COVID-19 deaths.
2. **Univariate Time-Series Analysis**: Employs decomposition methods to forecast the daily number of COVID-19 deaths in the United States.
3. **Deep Learning using LSTM**: Predicts COVID-19 deaths based on preceding death numbers, case counts, and cumulative vaccination numbers.

### Key Features
- **Feature Selection**: Lasso Regression helps identify significant predictors for COVID-19 deaths (e.g., mobility data, government stringency measures).
- **Univariate Forecasting**: Time-series decomposition and smoothing methods (SMA, EWMA, SES, TES) for trend analysis.
- **Deep Learning**: LSTM networks are optimized for COVID-19 death predictions, outperforming traditional RNNs.

### Project Structure

├── CovidRNN-Extloop.ipynb # Jupyter notebook with deep learning model and predictions ├── README.md # This file ├── requirements.txt # Project dependencies └── plots/ # Folder containing generated plots and figures


### Plots and Visualizations

1. **Feature Selection Using Lasso Regression**:
   - The importance of mobility features and government response is highlighted in the analysis. Key features include mobility to transit stations and government stringency measures.
   
   ![Feature Importance](plots/feature_importance.png)
   
   - Varying alpha values for feature selection are shown below:
   
   ![Alpha Coefficients](plots/alpha_coefficients.png)

2. **Time-Series Decomposition**:
   - Decomposition of COVID-19 deaths into seasonal, trend, and noise components. The multiplicative approach performed better, capturing the changing magnitude of seasonal components.

   ![Decomposition](plots/decomposition.png)

3. **LSTM Model Performance**:
   - LSTM models outperform traditional RNNs in terms of training and validation loss, as shown in the following plots:
   
   ![LSTM vs RNN](plots/lstm_vs_rnn.png)
   
   - Predictions improve when multiple features (deaths, cases, vaccinations) are considered:
   
   ![Multivariate Prediction](plots/multivariate_prediction.png)

### Installation

Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Dependencies include:
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras

### Usage
1. Clone the repository:
```git clone https://github.com/ConstantinEG17/CovidRNN-Project.git```
2. Navigate to the project directory:
```cd CovidRNN-Project```
3. Install dependencies:
```pip install -r requirements.txt```
4. Open the Jupyter notebook:
```jupyter notebook CovidRNN-Extloop.ipynb```
5. Run the cells in the notebook to execute the project.

### Results
Lasso Regression: Identified key features like mobility and government response stringency in predicting COVID-19 deaths.
Univariate Time-Series Analysis: While helpful for understanding trends, this method has limitations in capturing external factors like vaccination rates.
LSTM Performance: A multivariate approach using LSTM, with features such as COVID-19 cases and vaccinations, significantly improved prediction accuracy.

### Future Work
Feature Engineering: Adding more features such as hospitalization rates or public policy measures could improve prediction accuracy.
Model Optimization: Further tuning of hyperparameters and experimenting with different architectures could yield better results.
License
This project is licensed under the MIT License - see the LICENSE file for details.

### References
Google COVID-19 Open Data Repository
Worldometer COVID-19 Data



