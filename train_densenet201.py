"""
Script để train mô hình DenseNet-201 - Phiên bản cải tiến
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. Thiết lập tham số ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

TRAIN_DIR = "cropped_out/train"
TEST_DIR = "cropped_out/test"

# Tham số huấn luyện cải tiến
HEAD_EPOCHS = 15
FINETUNE_EPOCHS = 60  # Tăng lên 60
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-5
PATIENCE = 15  # Tăng patience lên 15

# --- 2. Tiền xử lý và Augmentation ---
def create_data_generators():
    """Tạo data generators với augmentation"""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Thêm augmentation độ sáng
        fill_mode='nearest',
        preprocessing_function=preprocess_input  # Chuẩn hóa theo ImageNet
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # Chuẩn hóa theo ImageNet
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, test_generator

# --- 3. Xây dựng mô hình DenseNet-201 ---
def build_densenet_model():
    """Xây dựng mô hình DenseNet-201 với custom head"""
    # Tải backbone
    base_model = DenseNet201(weights='imagenet', include_top=False,
                             input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Xây dựng head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Tăng dropout
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

# --- 4. Huấn luyện ---
def train_model():
    """Huấn luyện mô hình theo 2 giai đoạn với callbacks và learning rate đã tối ưu"""
    train_generator, test_generator = create_data_generators()
    model, base_model = build_densenet_model()

    # --- Giai đoạn 1: Huấn luyện Head ---
    print("\n--- Giai đoạn 1: Huấn luyện Head ---")
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=AdamW(learning_rate=HEAD_LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history_head = model.fit(
        train_generator,
        epochs=HEAD_EPOCHS,
        validation_data=test_generator
    )

    # --- Giai đoạn 2: Fine-tuning Backbone ---
    print("\n--- Giai đoạn 2: Fine-tuning Backbone ---")
    for layer in base_model.layers:
        if 'conv3_block' in layer.name or 'conv4_block' in layer.name or 'conv5_block' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer=AdamW(learning_rate=BACKBONE_LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Định nghĩa callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                   verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=PATIENCE // 2, verbose=1)
    model_checkpoint = ModelCheckpoint('densenet201_best_model.h5',
                                       monitor='val_accuracy', save_best_only=True, verbose=1)

    history_fine_tune = model.fit(
        train_generator,
        epochs=FINETUNE_EPOCHS,
        validation_data=test_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        initial_epoch=history_head.epoch[-1]
    )

    # --- 5. Lưu mô hình cuối cùng (đã được restore_best_weights) ---
    print("\n--- Lưu mô hình tốt nhất ---")
    # Không cần save lại vì EarlyStopping(restore_best_weights=True) và ModelCheckpoint đã xử lý
        print(f"Mô hình tốt nhất đã được lưu tại densenet201_best_model.h5")

if __name__ == "__main__":
    train_model()
