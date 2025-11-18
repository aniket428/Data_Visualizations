import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# -------------------------------
# Load Dataset
# -------------------------------
st.title("📊 Customer Churn Visualization Dashboard")

df = pd.read_csv("customer_churn_data.csv")

# Clean
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Preview", "Basic Plots", "Churn Analysis", "Correlation Heatmap"])

# -------------------------------
# Dataset preview
# -------------------------------
if section == "Dataset Preview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Column Information")
    st.write(df.info())

# -------------------------------
# Basic Plots
# -------------------------------
elif section == "Basic Plots":
    st.subheader("Basic Distributions")

    # Tenure
    st.write("### Tenure Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["tenure"], kde=True, ax=ax)
    st.pyplot(fig)

    # Monthly charges
    st.write("### Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["MonthlyCharges"], kde=True, ax=ax)
    st.pyplot(fig)

    # Payment method
    st.write("### Payment Method Count")
    fig, ax = plt.subplots()
    sns.countplot(x="PaymentMethod", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------
# Churn Analysis
# -------------------------------
elif section == "Churn Analysis":
    st.subheader("Churn Relationship Visualizations")

    # Churn count
    st.write("### Churn Count")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

    # Gender
    st.write("### Gender vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

    # Senior citizen
    st.write("### Senior Citizen vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="SeniorCitizen", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

    # Contract type
    st.write("### Contract Type vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

    # Internet Service
    st.write("### Internet Service vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="InternetService", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

# -------------------------------
# Heatmap
# -------------------------------
elif section == "Correlation Heatmap":
    st.subheader("Numeric Feature Correlation Heatmap")

    num_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]

    fig, ax = plt.subplots()
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

