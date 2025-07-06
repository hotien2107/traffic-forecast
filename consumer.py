# consumer_app.py
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import pyarrow

# --- Cấu hình ---
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'traffic_data'
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "traffic_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "traffic_features.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "traffic_label_encoder.pkl") # Thêm đường dẫn encoder
DATA_LAKE_PATH = "data/traffic_history"
CHECKPOINT_STORAGE = "checkpoints/storage"
CHECKPOINT_PREDICTION = "checkpoints/prediction"
# Sửa lại SEQUENCE_LENGTH cho khớp với file pretrain
SEQUENCE_LENGTH = 12

spark = SparkSession.builder \
    .appName("FinalTrafficConsumer_AI_Only_Fixed") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- Đọc nguồn Kafka ---
schema = StructType([
    StructField("time", StringType()), StructField("car", IntegerType()),
    StructField("motorbike", IntegerType()), StructField("bus", IntegerType()),
    StructField("truck", IntegerType()), StructField("total", IntegerType()),
    StructField("status", IntegerType()),
])
kafka_df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA_BROKER).option("subscribe", KAFKA_TOPIC).load()
df_with_ts = kafka_df.select(F.from_json(F.col("value").cast("string"), schema).alias("data")).select("data.*").withColumn("timestamp", F.col("time").cast(TimestampType()))

# --- LUỒNG 1: LƯU TRỮ ---
storage_query = df_with_ts.writeStream.format("parquet").outputMode("append").option("path", DATA_LAKE_PATH).option("checkpointLocation", CHECKPOINT_STORAGE).start()
print("✅ Luồng 1 (Lưu trữ) đã khởi động.")

# --- LUỒNG 2: DỰ ĐOÁN ---
# Gom dữ liệu thành cửa sổ
windowed_df = df_with_ts.groupBy(F.window("timestamp", f"{SEQUENCE_LENGTH * 15} seconds", "15 seconds")) \
    .agg(F.sort_array(F.collect_list(F.struct("timestamp", "car", "motorbike", "bus", "truck", "status"))).alias("sequence")) \
    .where(F.size(F.col("sequence")) == SEQUENCE_LENGTH)

# UDF Factory để dự đoán
def create_predict_udf():
    # Tải tất cả các thành phần cần thiết
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    features = pickle.load(open(FEATURES_PATH, 'rb'))
    label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
    model_weights = tf.keras.models.load_model(MODEL_PATH, compile=False).get_weights()

    ## [SỬA LẠI] ## - Đảm bảo kiến trúc model này GIỐNG HỆT trong file pretrain_lstm.py
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, len(features))),
        LSTM(64, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.set_weights(model_weights)
    print("Worker: Model AI đã được khởi tạo thành công với kiến trúc chính xác.")

    def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
        sequence_data = df['sequence'].iloc[0]

        # Thêm 'hour' và 'dayofweek' vào chuỗi dữ liệu
        processed_sequence = []
        for rec in sequence_data:
            ts = pd.to_datetime(rec['timestamp'])
            row = [rec[f] for f in ['car', 'motorbike', 'bus', 'truck']]
            row.extend([ts.hour, ts.dayofweek])
            processed_sequence.append(row)

        X_scaled = scaler.transform(np.array(processed_sequence, dtype=np.float32))

        # Dự đoán ra xác suất của các lớp
        prediction_probabilities = model.predict(np.array([X_scaled]), verbose=0)

        # Lấy ra lớp có xác suất cao nhất
        predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]

        # Chuyển chỉ số về lại nhãn gốc (ví dụ: "Ùn tắc")
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return pd.DataFrame([{"prediction_time": pd.Timestamp.now(), "predicted_status_label": predicted_label}])

    return predict_fn

# Áp dụng UDF
output_schema = StructType([StructField("prediction_time", TimestampType()), StructField("predicted_status_label", StringType())])
prediction_df = windowed_df.groupBy("window").applyInPandas(create_predict_udf(), schema=output_schema)

# Ghi ra console
prediction_query = prediction_df.writeStream.outputMode("update").format("console").option("truncate", "false").option("checkpointLocation", CHECKPOINT_PREDICTION).start()
print("✅ Luồng 2 (Dự đoán AI-Only) đã khởi động.")

spark.streams.awaitAnyTermination()