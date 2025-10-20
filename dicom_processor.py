#!/usr/bin/env python3
"""
DICOM to PNG Converter with Blue Annotation Removal
Converts DICOM files to PNG while removing blue annotations from brain scan area
"""

import cv2
import numpy as np
import pydicom
import os
import sys
import argparse


class DicomProcessor:
    def __init__(self):
        self.blue_lower = np.array([100, 50, 50])  # Lower bound for blue in HSV
        self.blue_upper = np.array([130, 255, 255])  # Upper bound for blue in HSV
        
    def read_dicom(self, dicom_path):
        """Read DICOM file and extract pixel data"""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            
            # Normalize pixel values to 0-255 range
            if pixel_array.dtype != np.uint8:
                pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
                pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to RGB if grayscale
            if len(pixel_array.shape) == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            
            return pixel_array
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            return None
    
    def enhance_image(self, image):
        """Enhance image quality using OpenCV"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def detect_brain_region(self, image):
        """Detect the central brain region to focus blue removal"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to find bright regions (brain tissue)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, use center region
            h, w = image.shape[:2]
            return (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
        
        # Find the largest contour (likely the brain)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand the bounding box slightly to ensure we cover the brain area
        margin = 50
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        return (x, y, x + w, y + h)
    
    def remove_blue_annotations(self, image):
        """Remove blue annotations from the brain region only"""
        # Get brain region coordinates
        x1, y1, x2, y2 = self.detect_brain_region(image)
        
        # Extract brain region
        brain_region = image[y1:y2, x1:x2].copy()
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(brain_region, cv2.COLOR_RGB2HSV)
        
        # Create mask for blue colors
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Inpaint the blue regions
        brain_cleaned = cv2.inpaint(brain_region, blue_mask, 3, cv2.INPAINT_TELEA)
        
        # Create result image by replacing only the brain region
        result = image.copy()
        result[y1:y2, x1:x2] = brain_cleaned
        
        return result
    
    def process_dicom(self, input_path, output_path=None):
        """Main processing function"""
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} does not exist")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}_processed.png"
        
        print(f"Processing DICOM file: {input_path}")
        
        # Read DICOM file
        image = self.read_dicom(input_path)
        if image is None:
            return False
        
        print("Enhancing image quality...")
        # Enhance image
        enhanced_image = self.enhance_image(image)
        
        print("Removing blue annotations from brain region...")
        # Remove blue annotations
        cleaned_image = self.remove_blue_annotations(enhanced_image)
        
        print(f"Saving processed image to: {output_path}")
        # Save as PNG using OpenCV
        # Convert RGB to BGR for OpenCV
        cleaned_image_bgr = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cleaned_image_bgr)
        
        print("Processing completed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Convert DICOM to PNG with blue annotation removal')
    parser.add_argument('input', help='Input DICOM file path')
    parser.add_argument('-o', '--output', help='Output PNG file path (optional)')
    
    args = parser.parse_args()
    
    processor = DicomProcessor()
    success = processor.process_dicom(args.input, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
