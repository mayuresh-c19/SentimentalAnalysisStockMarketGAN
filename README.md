# Stock Price Prediction using Generative Adversarial Networks (GANs)

This project aims to predict stock prices using Generative Adversarial Networks (GANs). GANs are a type of neural network architecture that consists of two networks, a generator and a discriminator, trained simultaneously to produce realistic synthetic data.

## Introduction

Stock price prediction is a challenging task due to the inherent volatility and complexity of financial markets. Traditional time-series forecasting methods often struggle to capture the non-linear and dynamic nature of stock price movements. In this project, we explore the use of GANs to generate synthetic stock price data and evaluate its performance against real-world data.

## Project Overview

The project consists of the following main components:

1. **Data Collection**: We collect historical stock price data and corresponding Twitter sentiment analysis data.
2. **Preprocessing**: The collected data is preprocessed to extract relevant features and prepare it for model training.
3. **Model Architecture**: We implement a GAN architecture consisting of a generator and a discriminator using TensorFlow and Keras.
4. **Training**: The GAN model is trained on the preprocessed data to learn the underlying patterns and relationships.
5. **Evaluation**: The trained model is evaluated on test data to assess its performance in predicting stock prices.
6. **Results Visualization**: We visualize the predicted stock prices and compare them with actual prices to gauge the model's effectiveness.

## Requirements

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- NLTK
- Statsmodels
- tqdm

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/stock-price-prediction.git
```

2. Navigate 
```bash
cd stock-price-prediction
```

3. Run Main script
```bash
python SentimentalAnalysis.py
```



