import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO
from kafka import KafkaProducer
import requests
from PIL import Image
import io
import os
import csv

URL_MAP = "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae7cfcbfd3d90017e8f422&camLocation=C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m%20%E2%80%93%20Ph%E1%BA%A1m%20V%C4%83n%20Hai&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
CAM_ID = "63ae7cfcbfd3d90017e8f422"
TARGET_CLASS_NAME = f"camImg-{CAM_ID}"

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'traffic_detection_topic'
CSV_FILE_PATH = 'traffic_log.csv'

YOLO_MODEL = YOLO('yolov8n.pt')
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorbike, bus, truck
WAIT_SECONDS = 15  # Thời gian chờ giữa các lần chạy

VEHICLE_CLASSES_MAP = {
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck'
}


def calculate_traffic_status(vehicle_counts: dict) -> int:
    weights = {'car': 1.0, 'motorbike': 0.4, 'bus': 2.5, 'truck': 2.5}
    congestion_score = sum(vehicle_counts.get(k, 0) * weights.get(k, 0) for k in vehicle_counts)

    if congestion_score < 5:
        return 1
    elif congestion_score < 10:
        return 2
    elif congestion_score < 15:
        return 3
    elif congestion_score < 20:
        return 4
    else:
        return 5


def save_data_to_csv(data_row: dict, file_path: str):
    """
    Lưu một dòng dữ liệu vào file CSV, tự động tạo header nếu file không tồn tại.
    """
    # Xử lý các giá trị lồng nhau để lưu vào CSV
    flat_data = data_row.copy()
    vehicle_details = flat_data.pop('vehicle_details', {})
    flat_data.update(vehicle_details)

    file_exists = os.path.isfile(file_path)
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            # Sắp xếp các trường để đảm bảo thứ tự cột nhất quán
            fieldnames = sorted(flat_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_data)
        print(f"[{datetime.now():%H:%M:%S}] [CSV] Đã lưu dữ liệu vào {file_path}")
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [CSV ERROR] Lỗi khi ghi file: {e}")


def count_vehicles_from_url(image_url: str) -> dict | None:
    """
    Tải ảnh từ URL, dùng YOLO để nhận diện và đếm số lượng cho từng loại phương tiện.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            results = YOLO_MODEL(image, verbose=False)

            vehicle_counts = {name: 0 for name in VEHICLE_CLASSES_MAP.values()}

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id in VEHICLE_CLASSES_MAP:
                    vehicle_name = VEHICLE_CLASSES_MAP[class_id]
                    vehicle_counts[vehicle_name] += 1

            return vehicle_counts

        except requests.exceptions.Timeout:
            print(
                f"[{datetime.now():%H:%M:%S}] [YOLO WARNING] Lỗi Timeout lần {attempt + 1}/{max_retries}. Đang thử lại...")
            time.sleep(5)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] [YOLO ERROR] Lỗi không xác định: {e}")
            break

    print(f"[{datetime.now():%H:%M:%S}] [YOLO ERROR] Không thể xử lý ảnh sau {max_retries} lần thử.")
    return None


def create_kafka_producer(broker_address: str):
    try:
        producer = KafkaProducer(
            bootstrap_servers=[broker_address],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8')
        )
        return producer
    except Exception as e:
        print(f"[KAFKA ERROR] Không thể kết nối: {e}")
        return None


def send_data_to_kafka(producer: KafkaProducer, topic: str, key: str, data: dict):
    try:
        future = producer.send(topic, key=key, value=data)
        future.get(timeout=10)
        print(f"[{datetime.now():%H:%M:%S}] [KAFKA] Đã gửi tin nhắn: {data}")
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [KAFKA ERROR] Lỗi khi gửi: {e}")


def pipeline():
    kafka_producer = create_kafka_producer(KAFKA_BROKER)
    if not kafka_producer:
        print("Không thể khởi tạo Kafka producer. Thoát chương trình.")
        return

    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(service=service, options=options)

    try:
        while True:
            print("\n" + "=" * 50)
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Bắt đầu chu trình mới...")

            image_url = None
            try:
                driver.get(URL_MAP)
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, TARGET_CLASS_NAME)))
                image_element = driver.find_element(By.CLASS_NAME, TARGET_CLASS_NAME)
                image_url = image_element.get_attribute('src')
                print(f"[{datetime.now():%H:%M:%S}] [CRAWLER] Lấy được URL ảnh: {image_url}")
            except Exception as e:
                print(f"[{datetime.now():%H:%M:%S}] [CRAWLER ERROR] Không lấy được URL: {e}")
                driver.refresh()

            if image_url:
                counts_by_type = count_vehicles_from_url(image_url)

                if counts_by_type is not None:
                    total_vehicles = sum(counts_by_type.values())

                    traffic_status = calculate_traffic_status(counts_by_type)

                    message_data = {
                        "timestamp_utc": datetime.utcnow().isoformat(),
                        "location": "CMT8_PhamVanHai",
                        "total_vehicles": total_vehicles,
                        "vehicle_details": counts_by_type,
                        "traffic_status": traffic_status,
                    }
                    send_data_to_kafka(kafka_producer, KAFKA_TOPIC, key=CAM_ID, data=message_data)

                    save_data_to_csv(message_data, CSV_FILE_PATH)

            print(f"[{datetime.now():%H:%M:%S}] Chu trình hoàn tất. Chờ {WAIT_SECONDS} giây...")
            time.sleep(WAIT_SECONDS)

    except KeyboardInterrupt:
        print("\n Dừng")
    finally:
        if kafka_producer:
            kafka_producer.close()
        driver.quit()


if __name__ == "__main__":
    pipeline()
