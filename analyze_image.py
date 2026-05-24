import cv2
import numpy as np
from PIL import Image
import sys

def analyze_image(image_path):
    """Analyze the image to understand its content and text visibility"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        print(f"Image loaded successfully: {image_path}")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Grayscale shape: {gray.shape}")
        print(f"Mean pixel value: {gray.mean():.2f}")
        print(f"Std pixel value: {gray.std():.2f}")

        # Check for text-like regions (high contrast areas)
        # Simple text detection heuristic
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Binary threshold: {_:.0f}")

        # Count white pixels (potential text)
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        text_ratio = white_pixels / total_pixels
        print(f"Text-like pixel ratio: {text_ratio:.4f}")

        # Check image dimensions
        height, width = gray.shape
        print(f"Image dimensions: {width}x{height}")

        # Estimate if this is a medical image
        if height > 500 and width > 500:
            print("Large image - likely contains detailed medical scan")
        else:
            print("Small image - may be cropped or thumbnail")

        # Check for very dark or very light regions
        dark_pixels = np.sum(gray < 50)
        light_pixels = np.sum(gray > 200)
        dark_ratio = dark_pixels / total_pixels
        light_ratio = light_pixels / total_pixels
        print(f"Dark pixel ratio (< 50): {dark_ratio:.4f}")
        print(f"Light pixel ratio (> 200): {light_ratio:.4f}")

    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        print("Usage: python analyze_image.py <image_path>")
