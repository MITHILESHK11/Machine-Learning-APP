import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.title("Simplified Machine Learning App")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Upload Data", "Data Visualization", "Train Model", "Make Predictions"])

if section == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())
        if st.checkbox("Show Dataset Summary"):
            st.write(data.describe())
            st.write("### Null Values:", data.isnull().sum())
        if st.checkbox("Clean Data"):
            data.fillna(data.mean(), inplace=True)
            st.write("Data cleaned!")
            st.write(data.head())
        st.session_state['data'] = data

if section == "Data Visualization":
    if 'data' in st.session_state:
        st.header("Data Visualization")
        data = st.session_state['data']
        plot_type = st.selectbox("Choose Plot Type", ["Scatter Plot", "Histogram", "Correlation Heatmap"])
        if plot_type == "Scatter Plot":
            x_col = st.selectbox("X-Axis", data.columns)
            y_col = st.selectbox("Y-Axis", data.columns)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x_col, y=y_col)
            st.pyplot(plt)
        elif plot_type == "Histogram":
            col = st.selectbox("Select Column", data.columns)
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True)
            st.pyplot(plt)
        elif plot_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)
    else:
        st.warning("Upload data first!")

if section == "Train Model":
    if 'data' in st.session_state:
        st.header("Train a Model")
        data = st.session_state['data']
        target_col = st.selectbox("Select Target Column", data.columns)
        X = data.drop(columns=[target_col])
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Logistic Regression", "Ridge", "Lasso",
                                                        "Decision Tree", "Random Forest"])
        if st.button("Train Model"):
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Ridge":
                model = Ridge()
            elif model_type == "Lasso":
                model = Lasso()
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                model = RandomForestClassifier()
            
            model.fit(X_train, y_train)
            if model_type in ["Linear Regression", "Ridge", "Lasso"]:
                y_pred = model.predict(X_test)
                st.write("### R² Score:", r2_score(y_test, y_pred))
            else:
                y_pred = model.predict(X_test)
                st.write("### Accuracy:", accuracy_score(y_test, y_pred))
                st.write("### Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
            joblib.dump(model, "trained_model.pkl")
            st.success("Model trained and saved!")
    else:
        st.warning("Upload data first!")

if section == "Make Predictions":
    st.header("Make Predictions")
    prediction_file = st.file_uploader("Upload new data for prediction", type=["csv", "xlsx"])
    if prediction_file:
        if prediction_file.name.endswith('.csv'):
            new_data = pd.read_csv(prediction_file)
        else:
            new_data = pd.read_excel(prediction_file)
        st.write("### New Data Preview")
        st.write(new_data.head())
        if st.button("Predict"):
            model = joblib.load("trained_model.pkl")
            predictions = model.predict(new_data)
            st.write("### Predictions:")
            st.write(predictions)
