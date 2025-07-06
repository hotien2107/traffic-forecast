import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

## 1. CẤU HÌNH
# ----------------------------------------------------
CSV_FILE_PATH = 'traffic_log.csv'
MODEL_DIR = "./models"  # Thư mục để lưu tất cả các thành phần

# Tên file cho các thành phần sẽ được lưu
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "traffic_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "traffic_label_encoder.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "traffic_features.pkl")

# Các tham số của mô hình
SEQUENCE_LENGTH = 12
TARGET_COLUMN = 'traffic_status'
FEATURE_COLUMNS = ['car', 'motorbike', 'bus', 'truck', 'hour', 'dayofweek']

## 2. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ----------------------------------------------------
print(f"Đang tải dữ liệu từ {CSV_FILE_PATH}...")
df = pd.read_csv(CSV_FILE_PATH)

# Chuyển đổi và sắp xếp theo thời gian
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df = df.sort_values('timestamp_utc').reset_index(drop=True)

# Feature Engineering: Thêm các đặc trưng về thời gian
df['hour'] = df['timestamp_utc'].dt.hour
df['dayofweek'] = df['timestamp_utc'].dt.dayofweek  # Thứ Hai=0, Chủ Nhật=6

label_encoder = LabelEncoder()
df['status_encoded'] = label_encoder.fit_transform(df[TARGET_COLUMN])

print("Dữ liệu sau khi xử lý:")
print(df[['timestamp_utc', 'traffic_status', 'status_encoded'] + FEATURE_COLUMNS].head())

## 3. CHUẨN BỊ DỮ LIỆU CHO MODEL
# ----------------------------------------------------
# Chuẩn hóa các feature
scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

# Tạo các chuỗi dữ liệu (sequences)
X, y = [], []
for i in range(len(df) - SEQUENCE_LENGTH):
    # Lấy ra một chuỗi các feature
    X.append(df.loc[i:i + SEQUENCE_LENGTH - 1, FEATURE_COLUMNS].values)
    # Lấy ra target tương ứng với chuỗi đó
    y.append(df.loc[i + SEQUENCE_LENGTH, 'status_encoded'])

X = np.array(X)
y = np.array(y)

print(f"Đã tạo {len(X)} chuỗi dữ liệu.")
print(f"Kích thước của X: {X.shape}")  # (số mẫu, độ dài chuỗi, số features)
print(f"Kích thước của y: {y.shape}")  # (số mẫu,)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

## 4. XÂY DỰNG VÀ HUẤN LUYỆN MODEL
# ----------------------------------------------------
print("\nBắt đầu xây dựng và huấn luyện mô hình LSTM...")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # (sequence_length, num_features)
    LSTM(64, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model cho bài toán phân loại
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Dùng loss này khi y là số nguyên
    metrics=['accuracy']
)

model.summary()

# Dừng sớm nếu không có cải thiện để tránh overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

print("\nHuấn luyện hoàn tất!")

## 5. LƯU TẤT CẢ CÁC THÀNH PHẦN
# ----------------------------------------------------
print(f"Đang lưu các thành phần vào thư mục '{MODEL_DIR}'...")
os.makedirs(MODEL_DIR, exist_ok=True)

# Lưu model
model.save(MODEL_PATH)
print(f"-> Đã lưu model tại: {MODEL_PATH}")

# Lưu scaler
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"-> Đã lưu scaler tại: {SCALER_PATH}")

# Lưu label encoder
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"-> Đã lưu label encoder tại: {LABEL_ENCODER_PATH}")

# Lưu danh sách các feature đã dùng
with open(FEATURES_PATH, 'wb') as f:
    pickle.dump(FEATURE_COLUMNS, f)
print(f"-> Đã lưu danh sách feature tại: {FEATURES_PATH}")

print("\n✅ Quá trình tiền huấn luyện hoàn tất. Hệ thống đã sẵn sàng để dự đoán.")
