"""
===============================================================================
SkyMetrics AI: מערכת ניתוח תנועה אווירית וזיהוי עומסים מבוססת Deep Learning
===============================================================================
פרויקט מקיף בראייה ממוחשבת ולמידה עמוקה הכולל:
1. Data Pipeline & Roboflow Integration
2. CNN & YOLOv8 Detection and Localization Training
3. Real-time Video Tracking & Anomaly/Traffic Analysis
4. Interactive Web Application (Streamlit Dashboard)
===============================================================================
"""

import os
import sys
import glob
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None
from ultralytics import YOLO

# =============================================================================
# 1. ניהול, הורדה ועיבוד מוקדם של ה-DATASET (Data Pipeline)
# =============================================================================

class AirTrafficDatasetManager:
    """
    מחלקה לניהול הורדת מאגר הנתונים מ-Roboflow, ניתוח המבנה הסטטיסטי,
    והצגת דוגמאות מתויגות מתוך התמונות (Data Preprocessing & Augmentation).
    """
    def __init__(self, api_key: str, workspace: str = "aircraft-detection", project_name: str = "planes", version: int = 1):
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name
        self.version = version
        self.dataset_location = None

    def download_dataset(self, model_format: str = "yolov8") -> str:
        """
        מתחבר ל-Roboflow API ומוריד את ה-Dataset.
        """
        print("[INFO] מתחבר ל-Roboflow API ומוריד את מאגר הנתונים...")
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project_name)
        dataset = project.version(self.version).download(model_format)
        self.dataset_location = dataset.location
        print(f"[SUCCESS] הנתונים הורדו והוכנו בנתיב: {self.dataset_location}")
        return self.dataset_location

    def get_dataset_stats(self) -> dict:
        """
        מחזיר ניתוח סטטיסטי של כמות התמונות והתגיות בחלוקה ל-Train, Valid, Test.
        """
        if not self.dataset_location or not os.path.exists(self.dataset_location):
            print("[WARN] לא נמצא נתיב Dataset תקף. אנא הורד את הנתונים תחילה.")
            return {}

        splits = ['train', 'valid', 'test']
        stats = {}

        for split in splits:
            img_dir = os.path.join(self.dataset_location, split, 'images')
            label_dir = os.path.join(self.dataset_location, split, 'labels')
            
            images = glob.glob(os.path.join(img_dir, '*.*'))
            labels = glob.glob(os.path.join(label_dir, '*.txt'))
            
            stats[split] = {
                'images_count': len(images),
                'labels_count': len(labels)
            }
            print(f"[STAT] {split.upper()} Set -> תמונות: {len(images)}, קובצי תגיות: {len(labels)}")
            
        return stats

    def visualize_sample_bounding_boxes(self, split: str = 'train', num_samples: int = 2):
        """
        מצייר ומציג תמונות לדוגמה מתוך ה-Dataset יחד עם ה-Bounding Boxes שחולצו מהתגיות.
        """
        if not self.dataset_location:
            return

        img_dir = os.path.join(self.dataset_location, split, 'images')
        label_dir = os.path.join(self.dataset_location, split, 'labels')
        image_files = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

        if not image_files:
            print("[WARN] לא נמצאו תמונות להצגה.")
            return

        fig, axes = plt.subplots(1, num_samples, figsize=(12, 6))
        if num_samples == 1:
            axes = [axes]

        for i in range(min(num_samples, len(image_files))):
            img_path = image_files[i]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    class_id, x_c, y_c, bw, bh = map(float, line.strip().split())
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Aircraft (Class {int(class_id)})", (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            axes[i].imshow(img)
            axes[i].set_title(f"Sample: {os.path.basename(img_path)}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# =============================================================================
# 2. בניית מודל ה-DEEP LEARNING, אימון והערכת ביצועים
# =============================================================================

class AirTrafficModelEngine:
    """
    מחלקה לבניית ארכיטקטורת ה-CNN/YOLO, הרצת תהליך האימון, והערכת המטריקות (mAP, Loss).
    """
    def __init__(self, model_version: str = "yolov8m.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] מנוע Deep Learning מורץ על גבי: {self.device}")
        self.model = YOLO(model_version)

    def train_model(self, data_yaml_path: str, epochs: int = 50, imgsz: int = 640, batch_size: int = 16):
        """
        אימון המודל תוך שימוש ב-Data Augmentation מורכב בלייב (Mosaic, Mixup, Scaling, Rotation).
        """
        print(f"[INFO] מתחיל אימון מודל עבור {epochs} אפוקים...")
        
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            name="airtraffic_yolo_run",
            project="runs/detect",
            
            # Data Augmentations
            degrees=15.0,
            translate=0.1,
            scale=0.2,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            
            # Optimization Settings
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            patience=10
        )
        print("[SUCCESS] תהליך האימון הושלם בהצלחה!")
        return results

    def plot_training_metrics(self, results_csv: str = "runs/detect/airtraffic_yolo_run/results.csv"):
        """
        טעינת קובץ המטריקות והצגת הגרפים של Precision, Recall, mAP50 ו-Loss.
        """
        if not os.path.exists(results_csv):
            print(f"[WARN] קובץ התוצאות לא נמצא בנתיב: {results_csv}")
            return

        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Box Loss
        axs[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
        axs[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red', linestyle='--')
        axs[0, 0].set_title('Localization Box Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Precision & Recall
        axs[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='green')
        axs[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange')
        axs[0, 1].set_title('Precision & Recall Curves')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # mAP Metrics
        axs[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP @ 0.50', color='purple')
        axs[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP @ 0.50:0.95', color='magenta', linestyle='--')
        axs[1, 0].set_title('Mean Average Precision (mAP)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Classification Loss
        axs[1, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='navy')
        axs[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='crimson', linestyle='--')
        axs[1, 1].set_title('Classification Loss')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

# =============================================================================
# 3. מעקב בווידאו וסיווג עומסים בזמן אמת (Video Tracking & Analytics)
# =============================================================================

class AirTrafficVideoProcessor:
    """
    מחלקה להרצת מעקב בזמן אמת (Object Tracking) על סרטוני וידאו של מסלולי תעופה,
    חישוב כמות כלי הטיס וזיהוי אנומליות/עומסים.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def process_video(self, input_video_path: str, output_video_path: str, congestion_threshold: int = 5):
        """
        מעבד סרטון וידאו פריים אחר פריים, משרטט Bounding Boxes, ומחשב רמת עומס.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"[ERROR] לא ניתן לפתוח את הקובץ: {input_video_path}")
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        out    = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"[INFO] מעבד וידאו: {input_video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # הרצת מעקב אובייקטים (Tracking)
            results = self.model.track(frame, persist=True)
            
            aircraft_count = len(results[0].boxes) if results[0].boxes else 0
            annotated_frame = results[0].plot()

            # הוספת שכבת מידע (Overlay) לניתוח עומס
            status_color = (0, 0, 255) if aircraft_count >= congestion_threshold else (0, 255, 0)
            status_str   = "HIGH CONGESTION" if aircraft_count >= congestion_threshold else "NORMAL TRAFFIC"
            
            overlay_text = f"Detected Aircraft: {aircraft_count} | Status: {status_str}"
            cv2.putText(annotated_frame, overlay_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"[SUCCESS] הווידאו המעובד נשמר בהצלחה ב: {output_video_path}")

# =============================================================================
# 4. ממשק משתמש אינטראקטיבי (Streamlit Interactive Dashboard)
# =============================================================================

def run_streamlit_dashboard():
    """
    מפעיל דשבורד אינטראקטיבי מלא ב-Streamlit עבור המשתמש.
    """
    import streamlit as st

    st.set_page_config(page_title="SkyMetrics AI Control Center", page_icon="🛫", layout="wide")

    st.title("🛫 SkyMetrics AI: מערכת לניתוח תנועה אווירית וזיהוי עומסים")
    st.markdown("""
    מערכת בלמידה עמוקה וראייה ממוחשבת (Computer Vision & Deep Learning) 
    המיועדת לזיהוי, איכון (Localization) וניתוח עומסי כלי טיס בתצלומי אוויר ולוויין בזמן אמת.
    """)

    st.sidebar.header("⚙️ הגדרות מודל")
    confidence = st.sidebar.slider("סף ביטחון הזיהוי (Confidence Threshold)", 0.1, 1.0, 0.4, 0.05)
    model_type = st.sidebar.selectbox("גרסת המודל", ["yolov8n.pt", "yolov8m.pt", "Custom Trained Weights"])

    @st.cache_resource
    def load_streamlit_model(weights_name):
        return YOLO(weights_name if weights_name != "Custom Trained Weights" else "yolov8n.pt")

    model = load_streamlit_model(model_type)

    uploaded_file = st.file_uploader("העלה תמונת אוויר / לוויין של שדה תעופה", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 תמונה מקורית")
            st.image(image, use_column_width=True)

        with st.spinner("מבצע איכון (Localization) וסיווג אובייקטים..."):
            img_np = np.array(image)
            results = model.predict(img_np, conf=confidence)
            res_plotted = results[0].plot()
            detected_count = len(results[0].boxes)

        with col2:
            st.subheader("🎯 תוצאת עיבוד ומציאת אובייקטים")
            st.image(res_plotted, use_column_width=True)

        st.markdown("---")
        st.subheader("📊 ניתוח מדדי עומס ושטח")

        m1, m2, m3 = st.columns(3)
        m1.metric(label="מספר מטוסים שזוהו", value=detected_count)
        m2.metric(label="רמת סף התראה", value="5 מטוסים")
        
        status_label = "עמוס (High Traffic)" if detected_count >= 5 else "תקין (Normal Traffic)"
        m3.metric(label="סטטוס עומס במסלול", value=status_label)

        if detected_count >= 5:
            st.error("⚠️ **התראת עומס:** זוהתה צפיפות גבוהה של כלי טיס באזור הנבדק! מומלץ לעדכן את בקרת המסלול.")
        else:
            st.success("✅ **תנועה סדירה:** עומס כלי הטיס באזור נמצא בטווח התקין.")

# =============================================================================
# 5. MAIN ENTRY POINT (נקודת הכניסה והרצה הראשית)
# =============================================================================

if __name__ == "__main__":
    # במידה והקוד מורכב דרך Streamlit (streamlit run main_project.py)
    if "streamlit" in sys.argv[0] or any("streamlit" in arg for arg in sys.argv):
        run_streamlit_dashboard()
    else:
        print("=================================================================")
        print("SkyMetrics AI - Execution Suite")
        print("=================================================================")
        print("1. להרצת הדשבורד האינטראקטיבי, הריצו בטרמינל:")
        print("   streamlit run main_project.py")
        print("-----------------------------------------------------------------")
        print("2. להרצת אימון קוד / ניתוח נתונים ישיר:")
        print("   חברו את ה-Roboflow API Key והפעילו את המחלקות בקובץ זה.")
        print("=================================================================")


@st.cache_resource
def load_streamlit_model(weights_name):
    # במידה והמשתמש בוחר Custom Trained Weights אך הקובץ אינו בשרת, נטען ברירת מחדל
    if weights_name == "Custom Trained Weights":
        if os.path.exists("runs/detect/airtraffic_yolo_run/weights/best.pt"):
            return YOLO("runs/detect/airtraffic_yolo_run/weights/best.pt")
        else:
            st.sidebar.warning("קובץ המשקלים המותאם לא נמצא, טוען מודל ברירת מחדל.")
            return YOLO("yolov8n.pt")
    return YOLO(weights_name)
