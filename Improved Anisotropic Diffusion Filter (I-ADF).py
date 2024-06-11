
import cv2
import numpy as np

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

# Load an image from LIDC-IDRI dataset (replace with actual path to dataset image)
img_path = 'path.jpg'

# Read the image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if img is None:
    print(f"Failed to load image from '{img_path}'. Please check the path.")
    exit()

# Apply anisotropic diffusion filter
filtered_img = anisotropic_diffusion(img)

# Display original and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save filtered image
cv2.imwrite('filtered_image.jpg', filtered_img)