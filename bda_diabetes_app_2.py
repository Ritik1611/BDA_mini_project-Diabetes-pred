import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kafka import KafkaConsumer
import json
import time
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

# ------------------------------------------------------------
# System setup (Python + Java)
# ------------------------------------------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["JAVA_HOME"] = r"D:\java\jdk-24"   # adjust if different
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

# ------------------------------------------------------------
# Streamlit UI setup
# ------------------------------------------------------------
st.set_page_config(page_title="Real-Time Diabetes Prediction", layout="centered")
st.title("🩺 Real-Time Diabetes Prediction Dashboard (PySpark + Kafka)")
st.write(
    "Streaming live patient data from **Kafka** and predicting diabetes using a trained **PySpark ML model**."
)

# ------------------------------------------------------------
# Start Spark Session
# ------------------------------------------------------------
@st.cache_resource
def init_spark():
    spark = (
        SparkSession.builder.appName("RealTimeDiabetes")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    return spark


spark = init_spark()

# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------
model_path = "best_model_pyspark"
if os.path.exists(model_path):
    best_model = PipelineModel.load(model_path)
    st.sidebar.success("✅ Loaded trained PySpark model.")
else:
    st.sidebar.error("❌ Model not found. Please train and save it as `best_model_pyspark` first.")
    st.stop()

# ------------------------------------------------------------
# Kafka Consumer Setup
# ------------------------------------------------------------
KAFKA_TOPIC = "diabetes_real_time"
KAFKA_BROKER = "localhost:9092"

try:
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="realtime-diabetes-group",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    st.sidebar.success(f"✅ Connected to Kafka topic: `{KAFKA_TOPIC}`")
except Exception as e:
    st.sidebar.error(f"❌ Kafka connection failed: {e}")
    st.stop()

# ------------------------------------------------------------
# Streamlit placeholders
# ------------------------------------------------------------
placeholder_input = st.empty()
placeholder_pred = st.empty()
chart_placeholder = st.empty()
prob_placeholder = st.empty()
stats_placeholder = st.empty()

records = []
st.sidebar.info("Listening for real-time patient data...")

# ------------------------------------------------------------
# Real-time Streaming Loop
# ------------------------------------------------------------
for message in consumer:
    try:
        patient_data = message.value  # incoming JSON from Kafka
        df = pd.DataFrame([patient_data])
        sdf = spark.createDataFrame(df)

        # Predict using PySpark model
        result = best_model.transform(sdf).select("prediction", "probability").collect()[0]
        prediction = int(result["prediction"])
        probability = float(result["probability"][1])

        # Append to history for visualization
        patient_data["Prediction"] = prediction
        patient_data["Probability"] = probability
        records.append(patient_data)
        df_vis = pd.DataFrame(records)

        # Display incoming data
        placeholder_input.subheader("📩 Incoming Patient Data")
        placeholder_input.json(patient_data)

        # Show prediction
        if prediction == 1:
            placeholder_pred.error(f"🔴 Diabetes predicted (Probability: {probability:.2f})")
        else:
            placeholder_pred.success(f"🟢 No Diabetes predicted (Probability: {probability:.2f})")

        # Live count chart
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Prediction", data=df_vis, palette=["lightgreen", "salmon"], ax=ax)
        ax.set_xticklabels(["No Diabetes", "Diabetes"])
        ax.set_title("Prediction Counts (Real-Time)")
        chart_placeholder.pyplot(fig)

        # Probability histogram
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.histplot(df_vis["Probability"], bins=20, color="skyblue", kde=True, ax=ax2)
        ax2.set_title("Prediction Probability Distribution")
        prob_placeholder.pyplot(fig2)

        # Stats summary
        total = len(df_vis)
        diabetes_count = df_vis["Prediction"].sum()
        no_diabetes_count = total - diabetes_count
        stats_placeholder.markdown(
            f"**Total Patients:** {total} | 🔴 **Predicted Diabetes:** {diabetes_count} | 🟢 **No Diabetes:** {no_diabetes_count}"
        )

        time.sleep(1)

    except Exception as e:
        st.error(f"⚠️ Error processing message: {e}")
        time.sleep(2)
