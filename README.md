# Indian-IPO-Deep-learning-classification-model
This project aims to build a deep learning classification model to predict whether an Indian Initial Public Offering (IPO) will list at a profit or not, based on historical market data.

# Project Overview
Leveraging a dataset of Indian IPOs, this project explores the data, preprocesses it, and builds a deep neural network using TensorFlow/Keras to predict the binary outcome of an IPO listing (profit or loss). The project follows a standard machine learning workflow including data exploration, cleaning, feature scaling, model building, training, and evaluation.

# Dataset
The project uses the "Indian IPO Market Data" dataset from Kaggel, which contains various features related to IPOs, including issue price, issue size, and subscription details from different investor categories. The target variable is derived from the listing gains percentage.

# Methodology
1. **Data Loading and Exploration:** The dataset is loaded and initial data analysis is performed to understand the data structure, identify key variables, and check for missing values.
2. **Feature Engineering:** A binary target variable (Listing_Gains_Profit) is created based on whether the Listing_Gains_Percent is greater than 0.
3. **Data Cleaning:** Outliers are identified using the Interquartile Range (IQR) method and treated by clipping values to within the defined bounds.
4. **Data Visualization:** Various plots (countplots, histograms, boxplots, scatterplots) are used to visualize the distribution of variables and relationships between them.
5. **Feature Scaling:** Predictor variables are scaled using normalization to ensure all features are within a similar range (0 to 1).
6. **Data Splitting:** The dataset is split into training and testing sets using a 70:30 ratio to evaluate the model's performance on unseen data.
7. **Model Architecture:** A deep neural network is built using TensorFlow's Keras Sequential API. The model consists of multiple dense layers with ReLU activation and a final output layer with sigmoid activation for binary classification. Dropout layers are included to mitigate overfitting.
8. **Model Training:** The model is compiled with the Adam optimizer and BinaryCrossentropy loss function. It is trained on the training data for 250 epochs.
9. **Model Evaluation:** The trained model's performance is evaluated on both the training and testing datasets using accuracy as the primary metric.
# Results
The trained deep learning model achieved an accuracy of approximately 75% on the training data and 74% on the test data. The consistent performance across both datasets suggests that the model generalizes reasonably well to unseen data.

# Requirements
- Python 3
- Jupyter Notebook or Google Colab
- pandas
- numpy
- seaborn
- matplotlib
- tensorflow
- scikit-learn
# Usage
To run this project:

1. Clone / Download the ipynb file.
2. Open the Jupyter Notebook or Colab file.
3. Ensure you have the required libraries installed.
4. Run the cells sequentially to execute the code.
