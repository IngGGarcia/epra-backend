"""
Gaussian Process Regression (GPR) Module

This module implements a Gaussian Process Regression model for predicting emotional responses
(valence or arousal) from vectorized image representations.

Justification for using Gaussian Process Regression:
1. Uncertainty Quantification: GPR provides not only predictions but also confidence intervals,
   which is crucial for understanding the reliability of emotional predictions.

2. Non-linear Relationships: GPR can capture complex, non-linear relationships between image
   features and emotional responses without requiring explicit feature engineering.

3. Small Dataset Performance: GPR performs well with small datasets (5+ samples), making it
   suitable for cases where collecting large amounts of labeled emotional data is challenging.

4. Probabilistic Framework: The probabilistic nature of GPR allows for better handling of
   uncertainty in emotional predictions, which is inherent in affective computing tasks.

5. Kernel Flexibility: The ability to use different kernel functions allows the model to
   adapt to various types of relationships between image features and emotional responses.

The module is designed to work with vectorized image representations as input features
and predict either valence or arousal values as the target variable.
"""
