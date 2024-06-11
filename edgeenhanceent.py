
import cv2
import numpy as np

# Function for Unsharp Masking Filter (UMF) for edge enhancement
def unsharp_masking(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Load the previously processed lung region extracted image (replace with actual path)
lung_region_path = 'lung_region_extracted.jpg'

# Read the lung region extracted image
lung_region = cv2.imread(lung_region_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if lung_region is None:
    print(f"Failed to load image from '{lung_region_path}'. Please check the path.")
    exit()

# Step 3: Apply Unsharp Masking Filter (UMF) for edge enhancement
enhanced_lung_region = unsharp_masking(lung_region)

# Display original lung region and enhanced lung region images
cv2.imshow('Lung Region Extracted', lung_region)
cv2.imshow('Enhanced Lung Region', enhanced_lung_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save enhanced lung region image
cv2.imwrite('enhanced_lung_region.jpg', enhanced_lung_region)