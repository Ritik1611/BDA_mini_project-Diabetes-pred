from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType, StructField, FloatType
from kafka import KafkaConsumer
import json

# ------------------------------
# 1. Initialize Spark Session
# ------------------------------
spark = SparkSession.builder \
    .appName("DiabetesPredictionKafkaConsumer") \
    .getOrCreate()

# ------------------------------
# 2. Load the saved PySpark pipeline
# ------------------------------
best_model = PipelineModel.load("best_model_pyspark")

# ------------------------------
# 3. Kafka consumer setup
# ------------------------------
consumer = KafkaConsumer(
    'diabetes-topic',  # your Kafka topic
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='diabetes-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# ------------------------------
# 4. Define the input schema
# ------------------------------
input_schema = StructType([
    StructField("Pregnancies", FloatType(), True),
    StructField("Glucose", FloatType(), True),
    StructField("BloodPressure", FloatType(), True),
    StructField("SkinThickness", FloatType(), True),
    StructField("Insulin", FloatType(), True),
    StructField("BMI", FloatType(), True),
    StructField("DiabetesPedigreeFunction", FloatType(), True),
    StructField("Age", FloatType(), True)
])

# ------------------------------
# 5. Prediction function
# ------------------------------
def predict_single(model, data_dict):
    """
    Takes a dictionary of patient data and predicts diabetes.
    """
    # Convert dictionary to Spark DataFrame
    df = spark.createDataFrame([data_dict], schema=input_schema)

    # Use pipeline model directly
    pred_row = model.transform(df).select("prediction", "probability").collect()[0]

    pred = int(pred_row['prediction'])
    prob = pred_row['probability'][pred]
    return pred, prob

# ------------------------------
# 6. Consume Kafka messages and predict
# ------------------------------
print("Listening for real-time patient data...")

for message in consumer:
    data = message.value
    try:
        prediction, probability = predict_single(best_model, data)
        print(f"Patient data: {data}")
        print(f"Prediction: {'Diabetic' if prediction==1 else 'Non-diabetic'}, Probability: {probability:.4f}")
        print("-" * 50)
    except Exception as e:
        print(f"Error processing data: {data}")
        print(e)
