import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ======================
# 1. Load Image
# ======================
img = cv2.imread("001.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# ======================
# 2. Nuclei Segmentation (Watershed Algorithm)
# ======================
# Contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# Thresholding and morphological operations
_, thresh = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Watershed preparation
sure_bg = cv2.dilate(opening, kernel, iterations=2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Watershed segmentation
_, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0
markers = cv2.watershed(img_rgb, markers)

# ======================
# 3. Feature Extraction for Nuclei
# ======================
nuclei_features = []
nuclei_contours = []

for label in np.unique(markers):
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
    circularity = 4 * np.pi * (area / (perimeter**2))
    mean_gray = cv2.mean(gray_enhanced, mask=mask)[0]
    nuclei_features.append([area, circularity, mean_gray])
    nuclei_contours.append(cnt)

# ======================
# 4. K-Means Clustering (Normal vs Abnormal)
# ======================
nuclei_features = np.array(nuclei_features)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(nuclei_features)

# Determine abnormal cell cluster
score0 = np.mean(nuclei_features[labels==0, 0]) * (1 - np.mean(nuclei_features[labels==0, 1]))
score1 = np.mean(nuclei_features[labels==1, 0]) * (1 - np.mean(nuclei_features[labels==1, 1]))
cancer_label = 0 if score0 > score1 else 1

# ======================
# 5. Visualization (Task 1 Core)
# ======================
final = img_rgb.copy()
# Draw abnormal cells with red contours
for i, cnt in enumerate(nuclei_contours):
    if labels[i] == cancer_label:
        cv2.drawContours(final, [cnt], -1, (255, 0, 0), 1)

# Show result
plt.figure(figsize=(10,6))
plt.imshow(final)
plt.title("Red contours = Abnormal / Malignant Cells")
plt.axis('off')
plt.show()