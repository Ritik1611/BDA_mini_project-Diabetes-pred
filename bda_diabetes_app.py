# bda_diabetes_app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector

# -------------------------------
# Initialize Spark Session
# -------------------------------
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("BDA Diabetes Prediction") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

spark = init_spark()

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("diabetes.csv"):
        df = pd.read_csv("diabetes.csv")
    else:
        st.error("No dataset found! Please upload diabetes.csv")
        st.stop()
    return df

# -------------------------------
# Convert Pandas to Spark DF
# -------------------------------
def pandas_to_spark(spark, df_pandas):
    return spark.createDataFrame(df_pandas)

# -------------------------------
# Data Preprocessing (PySpark)
# -------------------------------
def preprocess_data(sdf):
    # Replace zeros with mean for certain columns
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_to_fix:
        mean_val = sdf.selectExpr(f"avg({c}) as mean_{c}").collect()[0][0]
        sdf = sdf.withColumn(c, when(col(c) == 0, mean_val).otherwise(col(c)))
    return sdf

# -------------------------------
# Train Models (PySpark MLlib)
# -------------------------------
def train_models(sdf):
    feature_cols = [c for c in sdf.columns if c != "Outcome"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    train, test = sdf.randomSplit([0.8, 0.2], seed=42)

    # Logistic Regression
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="Outcome")
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    lr_model = lr_pipeline.fit(train)
    lr_preds = lr_model.transform(test)

    # Random Forest
    rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Outcome", numTrees=100)
    rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
    rf_model = rf_pipeline.fit(train)
    rf_preds = rf_model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="Outcome", metricName="areaUnderROC")
    lr_auc = evaluator.evaluate(lr_preds)
    rf_auc = evaluator.evaluate(rf_preds)

    best_model = rf_model if rf_auc > lr_auc else lr_model
    best_auc = max(lr_auc, rf_auc)

    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "best_model": best_model,
        "lr_auc": lr_auc,
        "rf_auc": rf_auc,
        "test_data": test
    }

# -------------------------------
# Evaluate and Plot
# -------------------------------
def plot_auc_comparison(lr_auc, rf_auc):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Logistic Regression", "Random Forest"], [lr_auc, rf_auc], color=["skyblue", "lightgreen"])
    ax.set_title("Model ROC-AUC Comparison")
    ax.set_ylim(0, 1)
    return fig

# -------------------------------
# Predict from user input
# -------------------------------
def predict_single(best_model, input_dict):
    df = pd.DataFrame([input_dict])
    sdf = pandas_to_spark(spark, df)
    assembler = VectorAssembler(inputCols=list(df.columns), outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    assembled = assembler.transform(sdf)
    scaled = scaler.fit(assembled).transform(assembled)
    preds = best_model.transform(scaled).select("prediction", "probability").collect()[0]
    return int(preds["prediction"]), float(preds["probability"][1])

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="BDA Diabetes Prediction", layout="centered")
st.title("🧠 Big Data Analytics — Diabetes Disease Prediction")
st.write("This project demonstrates **distributed data analytics and ML using PySpark** with an interactive **Streamlit UI**.")

# Sidebar options
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a section:", ["📊 EDA", "⚙️ Train Models", "🩺 Predict"])

uploaded_file = st.sidebar.file_uploader("Upload diabetes dataset (CSV)", type=["csv"])
df = load_data(uploaded_file)

# EDA Section
if section == "📊 EDA":
    st.subheader("Exploratory Data Analysis (Pandas level)")
    st.write("First five rows of dataset:")
    st.dataframe(df.head())

    st.write("Data summary:")
    st.write(df.describe())

    fig, ax = plt.subplots()
    sns.countplot(x="Outcome", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

# Train Models Section
elif section == "⚙️ Train Models":
    st.subheader("Train Models using PySpark MLlib")
    sdf = pandas_to_spark(spark, df)
    sdf = preprocess_data(sdf)

    with st.spinner("Training models on PySpark cluster..."):
        results = train_models(sdf)

    st.success("✅ Training complete!")
    st.write(f"**Logistic Regression AUC:** {results['lr_auc']:.4f}")
    st.write(f"**Random Forest AUC:** {results['rf_auc']:.4f}")
    fig = plot_auc_comparison(results['lr_auc'], results['rf_auc'])
    st.pyplot(fig)

    # Save best model
    best_model_path = "best_model_pyspark"
    results["best_model"].write().overwrite().save(best_model_path)
    st.write(f"💾 Best model saved to: `{best_model_path}`")

    st.session_state["best_model"] = results["best_model"]

# Predict Section
elif section == "🩺 Predict":
    st.subheader("Diabetes Prediction on New Input")

    st.write("Enter patient details below:")

    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
    bp = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
    skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.number_input("Insulin (mu U/ml)", 0, 1000, 79)
    bmi = st.number_input("BMI", 0.0, 100.0, 25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5, step=0.01)
    age = st.number_input("Age", 1, 120, 33)

    if st.button("Predict"):
        # Try loading saved model
        if os.path.exists("best_model_pyspark"):
            from pyspark.ml.pipeline import PipelineModel
            best_model = PipelineModel.load("best_model_pyspark")
        elif "best_model" in st.session_state:
            best_model = st.session_state["best_model"]
        else:
            st.error("Train the model first in the '⚙️ Train Models' section.")
            st.stop()

        input_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        with st.spinner("Predicting using trained PySpark model..."):
            pred, prob = predict_single(best_model, input_dict)

        if pred == 1:
            st.error(f"🔴 The model predicts **Diabetes (Probability: {prob:.2f})**")
        else:
            st.success(f"🟢 The model predicts **No Diabetes (Probability: {prob:.2f})**")

st.sidebar.markdown("---")
st.sidebar.info("Developed as a **Big Data Analytics Mini Project** using PySpark + Streamlit.")
