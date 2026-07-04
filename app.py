
from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    model = load(filename)
    y = model.predict(X)
    return y

def create_streamlit_app():
    st.title("Simple Regression model prediction")
    input_feature = st.slider("Input Feature for Prediction", min_value=-3.0, max_value=3.0, value=0.0)
    if st.button("Predict value"):
        prediction = load_and_predict([[input_feature]])
        st.write(f"Prediction: {prediction[0]}")
        visualize_difference(input_feature, prediction[0])

def visualize_difference(input_feature: float, prediction: ArrayLike):
    X = load("X.joblib")
    y = load("y.joblib")
    actual_target = y[_index_of_closest(X, input_feature)]
    difference = actual_target - prediction
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='grey', label='Dataset')
    plt.scatter(input_feature, actual_target, color='blue', zorder=5, label='Actual Target')
    plt.scatter(input_feature, prediction, color='red', zorder=5, label='Predicted Target')
    plt.legend()
    plt.title("Prediction vs Actual Target")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.grid(True)
    plt.plot([input_feature, input_feature], [actual_target, prediction], 'k--')
    plt.annotate(f"Difference = {difference:.2f}", xy=(input_feature, (actual_target + prediction) / 2), xytext=(input_feature + 0.2, (actual_target + prediction) / 2))
    st.pyplot(fig)

def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx

if __name__ == '__main__':
    create_streamlit_app()
