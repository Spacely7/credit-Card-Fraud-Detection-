# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- Title ---
st.title("🛡️ Credit Card Fraud Detection Dashboard")

# --- Load Dataset Automatically ---
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")  # Make sure this file is in the same folder as app.py
    return df

df = load_data()

st.success("Dataset Loaded Successfully!")

# --- Show Data ---
if st.checkbox("Show Raw Data"):
    st.write(df)

# --- Data Overview ---
st.subheader("Dataset Overview")
st.write("Number of transactions:", df.shape[0])
st.write("Number of features:", df.shape[1])

# Class distribution
st.subheader("Class Distribution")
class_counts = df['Class'].value_counts()
st.bar_chart(class_counts)

# --- Feature Correlation ---
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Data Preparation ---
st.sidebar.title("Model Training and Testing")
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

X = df.drop(["Class"], axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
)

# --- Model Selection ---
model_choice = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Random Forest"))

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()

# --- Train Model ---
if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Performance Metrics")

    # --- Metric Selection Dropdown ---
    metric_option = st.selectbox(
        "Select what you want to display:",
        ("Accuracy Only", "Classification Report", "Confusion Matrix", "All Metrics")
    )

    if metric_option == "Accuracy Only":
        st.metric(label="Accuracy", value=round(accuracy_score(y_test, y_pred), 4))
        st.metric(label="ROC-AUC", value=round(roc_auc_score(y_test, y_pred), 4))

    elif metric_option == "Classification Report":
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif metric_option == "Confusion Matrix":
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

    elif metric_option == "All Metrics":
        st.metric(label="Accuracy", value=round(accuracy_score(y_test, y_pred), 4))
        st.metric(label="ROC-AUC", value=round(roc_auc_score(y_test, y_pred), 4))

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

