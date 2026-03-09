1. Project Overview
This project focuses on the application of Deep Learning in medical imaging, specifically classifying brain ultrasound images to assist in early diagnosis. I utilized DenseNet121, a powerful Convolutional Neural Network (CNN) architecture, to handle the complexities and subtle features inherent in medical ultrasound data.

2. Why DenseNet121?
In medical imaging, datasets are often limited in size. I chose DenseNet121 over other architectures (like ResNet or VGG) for several technical reasons:

Feature Reuse: By connecting each layer to every other layer in a feed-forward fashion, the model encourages feature reuse, which is crucial for identifying fine patterns in ultrasound textures.

Parameter Efficiency: DenseNet requires fewer parameters than traditional CNNs, reducing the risk of overfitting on small-scale medical datasets.

Vanishing Gradient Mitigation: The dense connections ensure a smoother flow of gradients during backpropagation, leading to more stable training.

3. Key Technical Implementations
Transfer Learning: Leveraged pre-trained weights from ImageNet with strategic fine-tuning of the top layers to adapt the model to the grayscale/textural domain of ultrasound images.

Data Preprocessing: Implemented specialized filters to reduce speckle noise (common in ultrasound) and applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance structural details.

Handling Imbalance: Applied Class Weighting and Data Augmentation (Rotation, Zoom, Shearing) to ensure the model performs reliably across all diagnostic categories.

4. Evaluation Metrics
Instead of relying solely on Accuracy, this project evaluates performance based on:

Sensitivity (Recall): Ensuring no critical cases are missed.

Specificity: Minimizing false alarms.

Confusion Matrix & ROC-AUC Curve: Providing a transparent view of the model's diagnostic reliability.
