# bda_diabetes_app.py

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from kafka import KafkaProducer
from kafka import KafkaConsumer
import time

# Force PySpark to use same Python as driver
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ----------------------------
# Streamlit page config must be first Streamlit command
# ----------------------------
st.set_page_config(page_title="BDA Diabetes Prediction", layout="centered")
st.title("🧠 Big Data Analytics — Diabetes Disease Prediction")
st.write("This project demonstrates **distributed data analytics and ML using PySpark** with an interactive **Streamlit UI**.")

# ----------------------------
# Fix JAVA_HOME and PySpark Python versions
# ----------------------------
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

# Debug info
print("Python executable:", sys.executable)
print("JAVA_HOME =", os.environ["JAVA_HOME"])
os.system("java -version")

# -------------------------------
# PySpark imports
# -------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel

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
# Predict Single Input
# -------------------------------
def predict_single(best_model, input_dict):
    df_pandas = pd.DataFrame([input_dict])
    sdf = pandas_to_spark(spark, df_pandas)
    feature_cols = list(input_dict.keys())
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    sdf = assembler.transform(sdf)
    sdf = scaler.fit(sdf).transform(sdf)
    pred_row = best_model.transform(sdf).select("prediction", "probability").collect()[0]
    return int(pred_row["prediction"]), float(pred_row["probability"][1])

# -------------------------------
# Kafka Producer for real-time data
# -------------------------------
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a section:", ["📊 EDA", "⚙️ Train Models", "🩺 Predict"])

uploaded_file = st.sidebar.file_uploader("Upload diabetes dataset (CSV)", type=["csv"])
df = load_data(uploaded_file)

# -------------------------------
# EDA Section
# -------------------------------
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

# -------------------------------
# Train Models Section
# -------------------------------
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

# -------------------------------
# Predict Section
# -------------------------------
elif section == "🩺 Predict":
    st.subheader("Real-time Diabetes Prediction Dashboard")
    st.write("Patient data will be consumed from Kafka in real-time.")

    # Load trained model
    if os.path.exists("best_model_pyspark"):
        best_model = PipelineModel.load("best_model_pyspark")
    elif "best_model" in st.session_state:
        best_model = st.session_state["best_model"]
    else:
        st.error("Train the model first in the '⚙️ Train Models' section.")
        st.stop()

    # Kafka Consumer
    consumer = KafkaConsumer(
        'diabetes_real_time',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    # -------------------------------
    # Streamlit placeholders
    # -------------------------------
    placeholder_input = st.empty()
    placeholder_pred = st.empty()
    chart_placeholder = st.empty()
    prob_placeholder = st.empty()
    stats_placeholder = st.empty()

    # Data storage for visualization
    records = []

    # Real-time loop
    for message in consumer:
        input_dict = message.value

        # -------------------------------
        # Predict using the trained model
        # -------------------------------
        df_pandas = pd.DataFrame([input_dict])
        sdf = pandas_to_spark(spark, df_pandas)
        pred_row = best_model.transform(sdf).select("prediction", "probability").collect()[0]
        prediction = int(pred_row["prediction"])
        probability = float(pred_row["probability"][1])

        # Append record
        records.append({
            **input_dict,
            "prediction": prediction,
            "probability": probability
        })

        df_vis = pd.DataFrame(records)

        # -------------------------------
        # Display incoming data
        # -------------------------------
        placeholder_input.write("**Incoming Patient Data:**")
        placeholder_input.json(input_dict)

        if prediction == 1:
            placeholder_pred.error(f"🔴 Diabetes predicted (Probability: {probability:.2f})")
        else:
            placeholder_pred.success(f"🟢 No Diabetes predicted (Probability: {probability:.2f})")

        # -------------------------------
        # Update charts
        # -------------------------------
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="prediction", data=df_vis, palette=["lightgreen", "salmon"], ax=ax)
        ax.set_xticklabels(["No Diabetes", "Diabetes"])
        ax.set_title("Prediction Counts (Real-time)")
        chart_placeholder.pyplot(fig)

        # Probability histogram
        fig2, ax2 = plt.subplots(figsize=(6,3))
        sns.histplot(df_vis["probability"], bins=20, color="skyblue", kde=True, ax=ax2)
        ax2.set_title("Prediction Probability Distribution")
        prob_placeholder.pyplot(fig2)

        # Stats
        total = len(df_vis)
        diabetes_count = df_vis["prediction"].sum()
        no_diabetes_count = total - diabetes_count
        stats_placeholder.markdown(
            f"**Total Patients:** {total}  |  "
            f"**Predicted Diabetes:** {diabetes_count}  |  "
            f"**Predicted No Diabetes:** {no_diabetes_count}"
        )

        # Slow down loop a little to not overwhelm Streamlit
        time.sleep(0.1)

st.sidebar.markdown("---")
st.sidebar.info("Developed as a **Big Data Analytics Mini Project** using PySpark + Streamlit.")
