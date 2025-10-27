import json
import time
from kafka import KafkaProducer
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'diabetes_real_time'

def generate_patient_data():
    return {
        "Pregnancies": random.randint(0, 15),
        "Glucose": random.randint(50, 200),
        "BloodPressure": random.randint(40, 120),
        "SkinThickness": random.randint(10, 50),
        "Insulin": random.randint(15, 276),
        "BMI": round(random.uniform(18, 50), 1),
        "DiabetesPedigreeFunction": round(random.uniform(0.1, 2.5), 2),
        "Age": random.randint(21, 80)
    }

while True:
    patient = generate_patient_data()
    producer.send(topic, patient)
    print("Sent:", patient)
    time.sleep(5)
