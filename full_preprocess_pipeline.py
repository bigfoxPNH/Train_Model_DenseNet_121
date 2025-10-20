"""
Complete Skull Detection and Preprocessing Pipeline
This script integrates YOLOv8 skull detection with the existing preprocessing workflow
"""

import cv2
import numpy as np
import os
from pathlib import Path
from skull_detection_inference import SkullDetector
import matplotlib.pyplot as plt

class FlexibleSkullPreprocessor:
    def __init__(self, model_path="runs/detect/skull_detector/weights/best.pt"):
        """
        Initialize the flexible skull preprocessor with YOLOv8 model
        
        Args:
            model_path: Path to trained YOLOv8 skull detection model
        """
        self.skull_detector = SkullDetector(model_path)
        
    def detect_and_crop_skull(self, image_path, confidence_threshold=0.5, padding=20):
        """
        Detect and crop skull using YOLOv8 - the flexible solution
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detection
            padding: Extra pixels around bounding box
            
        Returns:
            numpy.ndarray: Cropped skull image or None if no detection
        """
        try:
            cropped_skull = self.skull_detector.crop_skull(
                image_path, 
                confidence_threshold=confidence_threshold,
                padding=padding
            )
            return cropped_skull
        except Exception as e:
            print(f"Error in skull detection: {e}")
            return None
    
    def enhance_contrast(self, image):
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def denoise_image(self, image):
        """
        Remove noise from the image
        """
        if len(image.shape) == 3:
            # For color images
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # For grayscale images
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def normalize_intensity(self, image):
        """
        Normalize image intensity to 0-255 range
        """
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def resize_image(self, image, target_size=(512, 512)):
        """
        Resize image to target size while maintaining aspect ratio
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def full_preprocessing_pipeline(self, image_path, output_dir="processed_output", 
                                  save_intermediate=True):
        """
        Complete preprocessing pipeline: detect, crop, enhance, and prepare skull image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save processed images
            save_intermediate: Whether to save intermediate processing steps
            
        Returns:
            dict: Processing results and file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image name
        image_name = Path(image_path).stem
        
        results = {
            'input_path': image_path,
            'image_name': image_name,
            'success': False,
            'steps': {}
        }
        
        print(f"\n=== Processing {image_name} ===")
        
        # Step 1: Detect and crop skull using YOLOv8
        print("1. Detecting and cropping skull...")
        cropped_skull = self.detect_and_crop_skull(image_path)
        
        if cropped_skull is None:
            print("❌ No skull detected in image")
            results['error'] = 'No skull detected'
            return results
        
        print("✓ Skull detected and cropped successfully")
        
        if save_intermediate:
            crop_path = os.path.join(output_dir, f"{image_name}_01_cropped.png")
            cv2.imwrite(crop_path, cropped_skull)
            results['steps']['cropped'] = crop_path
        
        # Step 2: Enhance contrast
        print("2. Enhancing contrast...")
        enhanced = self.enhance_contrast(cropped_skull)
        
        if save_intermediate:
            enhanced_path = os.path.join(output_dir, f"{image_name}_02_enhanced.png")
            cv2.imwrite(enhanced_path, enhanced)
            results['steps']['enhanced'] = enhanced_path
        
        # Step 3: Denoise
        print("3. Removing noise...")
        denoised = self.denoise_image(enhanced)
        
        if save_intermediate:
            denoised_path = os.path.join(output_dir, f"{image_name}_03_denoised.png")
            cv2.imwrite(denoised_path, denoised)
            results['steps']['denoised'] = denoised_path
        
        # Step 4: Normalize intensity
        print("4. Normalizing intensity...")
        normalized = self.normalize_intensity(denoised)
        
        if save_intermediate:
            normalized_path = os.path.join(output_dir, f"{image_name}_04_normalized.png")
            cv2.imwrite(normalized_path, normalized)
            results['steps']['normalized'] = normalized_path
        
        # Step 5: Resize to standard size
        print("5. Resizing to standard size...")
        resized = self.resize_image(normalized, target_size=(512, 512))
        
        # Save final result
        final_path = os.path.join(output_dir, f"{image_name}_final_processed.png")
        cv2.imwrite(final_path, resized)
        results['steps']['final'] = final_path
        
        print(f"✓ Processing complete! Final image: {final_path}")
        
        results['success'] = True
        results['final_image'] = resized
        results['final_path'] = final_path
        
        return results
    
    def process_batch(self, input_dir, output_dir="batch_processed"):
        """
        Process multiple images in batch
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        successful = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Processing {i+1}/{len(image_files)} ---")
            
            try:
                result = self.full_preprocessing_pipeline(str(image_path), output_dir)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_path': str(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n=== Batch Processing Summary ===")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {len(image_files) - successful}")
        
        return results

def demo_comparison():
    """
    Demo function to show the difference between traditional and YOLOv8 approach
    """
    print("=== Skull Detection Comparison Demo ===")
    print("Traditional approach: Fixed position, rigid algorithms")
    print("YOLOv8 approach: Flexible detection, works at any position/orientation")
    print("\nTo run the demo:")
    print("1. Train your YOLOv8 model first")
    print("2. Use FlexibleSkullPreprocessor for processing")
    print("3. Compare results with traditional methods")

if __name__ == "__main__":
    # Check if model exists
    model_path = "runs/detect/skull_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("❌ YOLOv8 model not found!")
        print(f"Expected location: {model_path}")
        print("\nTo get started:")
        print("1. Prepare your dataset using prepare_data.py")
        print("2. Label your images using LabelImg")
        print("3. Train the model using train_skull_detection.py")
        print("4. Then run this preprocessing pipeline")
        demo_comparison()
    else:
        print("✓ YOLOv8 model found!")
        print("Flexible skull preprocessing pipeline ready!")
        
        # Example usage
        print("\nExample usage:")
        print("preprocessor = FlexibleSkullPreprocessor()")
        print("result = preprocessor.full_preprocessing_pipeline('your_image.jpg')")
        print("# Or for batch processing:")
        print("results = preprocessor.process_batch('input_directory')")
