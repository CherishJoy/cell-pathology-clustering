import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ======================
# 1. Load Histopathological Image
# ======================
img = cv2.imread("001.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# ======================
# 2. Nuclei Segmentation Using Watershed Algorithm
# ======================
# Contrast enhancement with CLAHE to improve nuclei visibility
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# Thresholding and morphological operations to remove noise
_, thresh = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Prepare for Watershed segmentation (separate foreground/background)
sure_bg = cv2.dilate(opening, kernel, iterations=2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Execute Watershed segmentation to separate individual nuclei
_, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0
markers = cv2.watershed(img_rgb, markers)

# ======================
# 3. Feature Extraction for Nuclei
# Extract key features: area, circularity, mean gray value
# ======================
nuclei_features = []
nuclei_contours = []

for label in np.unique(markers):
    # Skip background/unknown labels
    if label in (0, -1, 1):
        continue
    mask = (markers == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    # Calculate circularity (1 = perfect circle, <1 = irregular shape)
    circularity = 4 * np.pi * (area / (perimeter**2))
    # Calculate mean gray value of the nucleus
    mean_gray = cv2.mean(gray_enhanced, mask=mask)[0]
    nuclei_features.append([area, circularity, mean_gray])
    nuclei_contours.append(cnt)

# ======================
# 4. K-Means Clustering to Classify Normal/Abnormal Cells
# ======================
nuclei_features = np.array(nuclei_features)
# Use K-Means with 2 clusters (normal vs abnormal)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(nuclei_features)

# Determine which cluster represents abnormal (malignant) cells
# Abnormal cells typically have larger area and lower circularity
score0 = np.mean(nuclei_features[labels==0, 0]) * (1 - np.mean(nuclei_features[labels==0, 1]))
score1 = np.mean(nuclei_features[labels==1, 0]) * (1 - np.mean(nuclei_features[labels==1, 1]))
cancer_label = 0 if score0 > score1 else 1

# ======================
# 5. Task 2: Cell Counting (Core New Logic)
# Count normal/abnormal cells based on clustering results
# ======================
# Calculate cell counts
abnormal_count = np.sum(labels == cancer_label)
normal_count = len(labels) - abnormal_count
total_count = len(labels)

# Print counting results (console output for Task 2)
print("===== Cell Counting Results (Task 2) =====")
print(f"Total cells detected: {total_count}")
print(f"Normal cells: {normal_count}")
print(f"Abnormal (malignant) cells: {abnormal_count}")
print(f"Abnormal cell ratio: {abnormal_count/total_count:.2%}")

# ======================
# 6. Visualization (Task 1 + Task 2 Combined Annotation)
# Red contours = abnormal cells, Green contours = normal cells
# ======================
final = img_rgb.copy()
# Draw contours for normal/abnormal cells
for i, cnt in enumerate(nuclei_contours):
    if labels[i] == cancer_label:
        # Red contours for abnormal (malignant) cells (RGB: 255,0,0)
        cv2.drawContours(final, [cnt], -1, (255, 0, 0), 1)
    else:
        # Green contours for normal cells (RGB: 0,255,0)
        cv2.drawContours(final, [cnt], -1, (0, 255, 0), 1)

# Add text annotations for cell counts on the image
cv2.putText(final, f"Total: {total_count}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(final, f"Normal: {normal_count}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(final, f"Abnormal: {abnormal_count}", (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Save the final result image with counting annotations
plt.figure(figsize=(12,8))
plt.imshow(final)
plt.title("Cell Clustering + Counting Result (Task 1 + Task 2)")
plt.axis('off')
plt.savefig("result_with_count.png", dpi=300, bbox_inches='tight')  # Save result with counts
plt.show()

# Additional console output to confirm image saving
print("\nResult image saved as 'result_with_count.png' (includes counting annotations)")