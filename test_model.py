"""
Script để test và đánh giá hiệu suất mô hình DenseNet-121 đã huấn luyện
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Thiết lập tham số ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

TEST_DIR = "cropped_out/test"
MODEL_PATH = "densenet121_best_model.h5"

# --- 2. Tạo data generator cho test ---
def create_test_generator():
    """Tạo test data generator với cùng preprocessing như khi train"""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # Không shuffle để có thể map predictions với labels
    )
    
    return test_generator

# --- 3. Load mô hình đã huấn luyện ---
def load_trained_model():
    """Load mô hình tốt nhất đã được lưu"""
    print(f"Đang load mô hình từ {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Mô hình đã được load thành công!")
    return model

# --- 4. Đánh giá mô hình ---
def evaluate_model():
    """Đánh giá chi tiết hiệu suất mô hình"""
    
    # Load mô hình và tạo test generator
    model = load_trained_model()
    test_generator = create_test_generator()
    
    print(f"\n--- THÔNG TIN TẬP TEST ---")
    print(f"Tổng số ảnh test: {test_generator.samples}")
    print(f"Số classes: {test_generator.num_classes}")
    print(f"Class indices: {test_generator.class_indices}")
    
    # Đánh giá trên toàn bộ tập test
    print(f"\n--- ĐÁNH GIÁ TỔNG QUAN ---")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Dự đoán chi tiết
    print(f"\n--- DỰ ĐOÁN CHI TIẾT ---")
    predictions = model.predict(test_generator, verbose=1)
    
    # Chuyển đổi predictions thành class labels
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Lấy true labels
    true_classes = test_generator.classes
    
    # Tạo classification report
    class_names = list(test_generator.class_indices.keys())
    print(f"\n--- BÁO CÁO PHÂN LOẠI ---")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_names, digits=4))
    
    # Tạo confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print(f"\n--- CONFUSION MATRIX ---")
    print(cm)
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix đã được lưu tại: confusion_matrix.png")
    
    # Thống kê chi tiết theo từng class
    print(f"\n--- THỐNG KÊ CHI TIẾT THEO CLASS ---")
    for i, class_name in enumerate(class_names):
        class_mask = (true_classes == i)
        class_predictions = predicted_classes[class_mask]
        class_true = true_classes[class_mask]
        
        if len(class_true) > 0:
            class_accuracy = np.mean(class_predictions == class_true)
            print(f"{class_name}:")
            print(f"  - Số lượng: {len(class_true)}")
            print(f"  - Độ chính xác: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
            print(f"  - Dự đoán đúng: {np.sum(class_predictions == class_true)}/{len(class_true)}")
    
    # Phân tích confidence scores
    print(f"\n--- PHÂN TÍCH CONFIDENCE SCORES ---")
    print(f"Confidence trung bình: {np.mean(predictions):.4f}")
    print(f"Confidence cao nhất: {np.max(predictions):.4f}")
    print(f"Confidence thấp nhất: {np.min(predictions):.4f}")
    print(f"Độ lệch chuẩn: {np.std(predictions):.4f}")
    
    # Đếm số predictions với confidence cao/thấp
    high_confidence = np.sum((predictions > 0.8) | (predictions < 0.2))
    medium_confidence = np.sum((predictions >= 0.2) & (predictions <= 0.8))
    
    print(f"\nPhân bố confidence:")
    print(f"  - High confidence (>0.8 hoặc <0.2): {high_confidence} ({high_confidence/len(predictions)*100:.1f}%)")
    print(f"  - Medium confidence (0.2-0.8): {medium_confidence} ({medium_confidence/len(predictions)*100:.1f}%)")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes,
        'class_names': class_names
    }

# --- 5. Chạy đánh giá ---
if __name__ == "__main__":
    print("=== ĐÁNH GIÁ MÔ HÌNH DENSENET-121 ===")
    
    try:
        results = evaluate_model()
        print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
        print(f"Độ chính xác trên tập test: {results['test_accuracy']*100:.2f}%")
        print(f"Loss trên tập test: {results['test_loss']:.4f}")
        
    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình: {str(e)}")
        print("Vui lòng kiểm tra:")
        print("1. File mô hình tồn tại tại đường dẫn đã chỉ định")
        print("2. Thư mục test data có đúng cấu trúc")
        print("3. Các thư viện cần thiết đã được cài đặt")
