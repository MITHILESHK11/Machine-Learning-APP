import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page configuration
st.set_page_config(page_title="AutoML Pipeline", layout="wide")

# Title
st.title("Fully Automated Machine Learning Pipeline")
st.write("Upload your dataset, and let the app handle everything from preprocessing to deployment!")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Function to preprocess data
def preprocess_data(df, target):
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y

# Function for feature engineering
def feature_engineering(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

# Function to select and train the best model
def train_model(X, y, problem_type):
    if problem_type == "classification":
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    else:
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if problem_type == "classification" else 'neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    return best_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == "classification":
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
    else:
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Main app logic
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select target variable
    target = st.selectbox("Select the target variable", df.columns)
    
    # Infer problem type
    if df[target].nunique() <= 10:  # Arbitrary threshold for classification
        problem_type = "classification"
    else:
        problem_type = "regression"
    st.write(f"### Problem Type: {problem_type.capitalize()}")

    # Preprocess data
    st.write("### Data Preprocessing")
    X, y = preprocess_data(df, target)
    st.write("Preprocessed Data:")
    st.write(X.head())

    # Feature engineering
    st.write("### Feature Engineering")
    X = feature_engineering(X)
    st.write("Engineered Features Shape:", X.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    st.write("### Model Training")
    model = train_model(X_train, y_train, problem_type)
    st.write("Best Model:", model)

    # Evaluate model
    st.write("### Model Evaluation")
    evaluate_model(model, X_test, y_test, problem_type)

    # Cross-validation
    st.write("### Cross-Validation")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if problem_type == "classification" else 'neg_mean_squared_error')
    st.write("Cross-Validation Scores:", cv_scores)
    st.write("Mean CV Score:", np.mean(cv_scores))

    # Save model
    st.write("### Model Deployment")
    model_filename = "trained_model.pkl"
    joblib.dump(model, model_filename)
    with open(model_filename, "rb") as f:
        st.download_button("Download Trained Model", f, file_name=model_filename)

    # Monitoring and maintenance (simulated)
    st.write("### Monitoring and Maintenance")
    st.write("Model performance logs saved for future monitoring.")
    with open("model_logs.txt", "a") as f:
        f.write(f"{time.ctime()}: Model trained with CV score {np.mean(cv_scores)}\n")

else:
    st.write("Please upload a dataset to get started.")

# Data Visualization
if uploaded_file is not None:
    st.write("### Data Visualization")
    st.write("#### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.write("#### Pairplot")
    sns.pairplot(df)
    st.pyplot(plt)

# Footer
st.write("---")
st.write("Built with ❤️ using Streamlit")
