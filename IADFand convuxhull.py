
import cv2
import numpy as np

# Function for noise removal using Improved Anisotropic Diffusion Filter (I-ADF)
def anisotropic_diffusion(img, iterations=10, delta=0.15, kappa=30):
    img = np.float32(img)
    img_filtered = img.copy()
    for _ in range(iterations):
        dN = img_filtered[:-2, 1:-1] - img_filtered[1:-1, 1:-1]
        dS = img_filtered[2:, 1:-1] - img_filtered[1:-1, 1:-1]
        dE = img_filtered[1:-1, :-2] - img_filtered[1:-1, 1:-1]
        dW = img_filtered[1:-1, 2:] - img_filtered[1:-1, 1:-1]

        cN = np.exp(-(dN / kappa)**2)
        cS = np.exp(-(dS / kappa)**2)
        cE = np.exp(-(dE / kappa)**2)
        cW = np.exp(-(dW / kappa)**2)

        img_filtered[1:-1, 1:-1] += delta * (cN*dN + cS*dS + cE*dE + cW*dW)

    return np.uint8(img_filtered)

# Function to extract convex hull of lung region
def extract_lung_region(img):
    # Convert image to binary (assuming lung region is already separated)
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull for the largest contour (lung region)
    if contours:
        lung_contour = max(contours, key=cv2.contourArea)
        convex_hull = cv2.convexHull(lung_contour)
        
        # Create a mask for the convex hull
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [convex_hull], -1, (255), thickness=cv2.FILLED)
        
        # Apply mask to extract lung region
        lung_region = cv2.bitwise_and(img, mask)
        
        return lung_region
    else:
        print("No contours found.")
        return img

# Load an image from LIDC-IDRI dataset (replace with actual path to dataset image)
img_path = 'path.jpg'

# Read the image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if img is None:
    print(f"Failed to load image from '{img_path}'. Please check the path.")
    exit()

# Step 1: Apply anisotropic diffusion for noise removal
filtered_img = anisotropic_diffusion(img)

# Step 2: Extract lung region using convex hull
lung_region = extract_lung_region(filtered_img)

# Display original, filtered, and lung region extracted images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.imshow('Lung Region Extracted', lung_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save filtered image and lung region extracted image
cv2.imwrite('filtered_image.jpg', filtered_img)
cv2.imwrite('lung_region_extracted.jpg', lung_region)