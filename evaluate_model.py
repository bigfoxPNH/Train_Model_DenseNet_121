"""
Script đánh giá hiệu suất model DenseNet121 đã được huấn luyện
Tính toán các chỉ số: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1-score
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K

# Định nghĩa Focal Loss (cần thiết để load model)
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = K.binary_crossentropy(y_true, y_pred)

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    modulating_factor = K.pow(1.0 - p_t, gamma)

    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    focal_loss = alpha_factor * modulating_factor * bce

    return K.mean(focal_loss, axis=-1)

# Thiết lập đường dẫn
TEST_DIR = 'data/test'
MODEL_PATH = 'densenet121_best_model.h5'

# Thông số
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def load_and_evaluate_model():
    """Load model và đánh giá trên test set"""
    
    print("=== ĐÁNH GIÁ MODEL DENSENET121 ===")
    print(f"Đang load model từ: {MODEL_PATH}")
    
    # Load model với custom objects
    model = load_model(MODEL_PATH, custom_objects={
        'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy
    })
    
    print("Model đã được load thành công!")
    
    # Chuẩn bị test data
    print("Đang chuẩn bị test data...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # Quan trọng: không shuffle để giữ thứ tự
    )
    
    print(f"Tìm thấy {test_generator.samples} ảnh test thuộc {len(test_generator.class_indices)} classes")
    print(f"Class indices: {test_generator.class_indices}")
    
    # Dự đoán
    print("Đang thực hiện dự đoán...")
    predictions = model.predict(test_generator, verbose=1)
    
    # Lấy true labels
    y_true = test_generator.classes
    y_pred_proba = predictions.flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Tính toán các metrics
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # PR-AUC
    pr_auc = average_precision_score(y_true, y_pred_proba)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision
    precision = precision_score(y_true, y_pred)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    
    # Recall
    recall = recall_score(y_true, y_pred)
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    
    # F1-score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-score: {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Normal', 'Pneumonia']))
    
    # Tóm tắt kết quả theo format yêu cầu
    print(f"\n=== TÓM TẮT KẾT QUẢ ===")
    print(f"ROC-AUC     {roc_auc*1000:.0f}")
    print(f"PR-AUC      {pr_auc*1000:.0f}")
    print(f"Accuracy    {accuracy*100:.1f}%")
    print(f"Precision   {precision*100:.1f}%")
    print(f"Recall      {recall*100:.1f}%")
    print(f"F1-score    {f1*100:.1f}%")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

if __name__ == "__main__":
    try:
        results = load_and_evaluate_model()
        print("\nĐánh giá hoàn tất!")
        
    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá: {str(e)}")
        print("Vui lòng kiểm tra:")
        print("1. File model tồn tại tại đường dẫn đã chỉ định")
        print("2. Thư mục test data tồn tại và có cấu trúc đúng")
        print("3. Các thư viện cần thiết đã được cài đặt")
