"""
Model Evaluation Script for DenseNet121 Skull Classification
This script evaluates the trained model and calculates comprehensive metrics:
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- Accuracy
- Precision
- Recall
- F1-score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to the trained model (.h5 file)
            test_data_path: Path to test data directory
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.test_generator = None
        self.predictions = None
        self.true_labels = None
        self.class_names = None
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
    def prepare_test_data(self, img_size=(224, 224), batch_size=32):
        """Prepare test data generator"""
        print(f"Preparing test data from: {self.test_data_path}")

        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.test_generator = test_datagen.flow_from_directory(
            self.test_data_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',  # Changed to binary for binary classification
            shuffle=False,  # Important: don't shuffle for evaluation
            classes=['Qualified_PNG', 'Unqualified_PNG']  # Explicitly specify classes
        )

        self.class_names = ['Qualified_PNG', 'Unqualified_PNG']
        print(f"Found {self.test_generator.samples} test images")
        print(f"Classes: {self.class_names}")
        
    def make_predictions(self):
        """Make predictions on test data"""
        print("Making predictions on test data...")
        
        # Get predictions
        self.predictions = self.model.predict(self.test_generator, verbose=1)
        
        # Get true labels
        self.true_labels = self.test_generator.classes
        
        print(f"Predictions shape: {self.predictions.shape}")
        print(f"True labels shape: {self.true_labels.shape}")
        
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        print("\n" + "="*50)
        print("CALCULATING EVALUATION METRICS")
        print("="*50)

        # For binary classification with sigmoid output
        y_pred_proba = self.predictions.flatten()  # Flatten to 1D array
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        y_true_binary = self.true_labels

        # Calculate metrics
        metrics = {
            'ROC-AUC': roc_auc_score(y_true_binary, y_pred_proba),
            'PR-AUC': average_precision_score(y_true_binary, y_pred_proba),
            'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'Precision': precision_score(y_true_binary, y_pred_binary),
            'Recall': recall_score(y_true_binary, y_pred_binary),
            'F1-score': f1_score(y_true_binary, y_pred_binary)
        }

        # Print metrics
        print("\nEVALUATION RESULTS:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name:12}: {value:.4f}")

        return metrics
        
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        y_pred_classes = (self.predictions.flatten() > 0.5).astype(int)
        cm = confusion_matrix(self.true_labels, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        # Binary classification with sigmoid output
        y_pred_proba = self.predictions.flatten()
        fpr, tpr, _ = roc_curve(self.true_labels, y_pred_proba)
        auc_score = roc_auc_score(self.true_labels, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")

        plt.show()
            
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve"""
        # Binary classification with sigmoid output
        y_pred_proba = self.predictions.flatten()
        precision, recall, _ = precision_recall_curve(self.true_labels, y_pred_proba)
        pr_auc = average_precision_score(self.true_labels, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to: {save_path}")

        plt.show()
    
    def generate_detailed_report(self, save_path=None):
        """Generate detailed classification report"""
        y_pred_classes = (self.predictions.flatten() > 0.5).astype(int)
        report = classification_report(self.true_labels, y_pred_classes, 
                                     target_names=self.class_names)
        
        print("\nDETAILED CLASSIFICATION REPORT:")
        print("=" * 50)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("DETAILED CLASSIFICATION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(report)
            print(f"Detailed report saved to: {save_path}")
        
        return report
    
    def run_complete_evaluation(self, save_plots=True, output_dir="evaluation_results"):
        """Run complete evaluation pipeline"""
        print("STARTING COMPLETE MODEL EVALUATION")
        print("=" * 60)
        
        # Create output directory
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load model and prepare data
        self.load_model()
        self.prepare_test_data()
        
        # Make predictions
        self.make_predictions()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate plots and reports
        if save_plots:
            self.plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"))
            self.plot_roc_curve(os.path.join(output_dir, "roc_curve.png"))
            self.plot_precision_recall_curve(os.path.join(output_dir, "pr_curve.png"))
            self.generate_detailed_report(os.path.join(output_dir, "detailed_report.txt"))
        else:
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.generate_detailed_report()
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        return metrics

def main():
    """Main function to run the evaluation"""
    # Configuration
    MODEL_PATH = "../../densenet121_best_model.h5"  # Path to your trained model
    TEST_DATA_PATH = "."  # Current directory (test folder)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check the model path and try again.")
        return
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data directory not found at {TEST_DATA_PATH}")
        print("Please check the test data path and try again.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(MODEL_PATH, TEST_DATA_PATH)
    
    # Run complete evaluation
    try:
        metrics = evaluator.run_complete_evaluation()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv("evaluation_results/metrics_summary.csv", index=False)
        print("Metrics summary saved to: evaluation_results/metrics_summary.csv")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your model and data paths.")

if __name__ == "__main__":
    main()
