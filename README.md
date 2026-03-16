# Cell Pathology Analysis for AI Oncology Assessment
This repository is developed to complete an assessment assignment for AI Oncology, which includes two core tasks for histopathological image analysis (Task 1: Cell Clustering & Annotation; Task 2: Cell Counting).

## Assignment Tasks & Implementation
### Task 1: Cell Clustering & Annotation
- **Requirement**: Perform unsupervised clustering on cells in histopathological images to distinguish normal/abnormal (malignant) cells, and submit the annotated image with abnormal cells highlighted.
- **Implementation**: Combined Watershed algorithm (nuclei segmentation) + K-Means clustering (cell classification). Abnormal cells are marked with red contours in the result image.
- **Result File**: `result.png` (Task 1 basic annotation)

### Task 2: Cell Counting & Annotated Statistics
- **Requirement**: Count the number of different types of cells (normal/abnormal) and provide annotated statistics.
- **Implementation**: Based on Task 1's clustering results, add quantitative counting logic and text annotation on the result image (all code comments and output are in English).
- **Key Results**:
  - Normal cells: Calculated automatically (green contours in the image)
  - Abnormal (malignant) cells: Calculated automatically (red contours in the image)
  - Total cells: Calculated automatically
  - Abnormal cell ratio: Calculated and displayed in console/output
- **Result File**: `result_with_count.png` (Task 2 counting + annotation with English text)

## Technical Stack
- Image Processing: OpenCV (Watershed segmentation, contour detection, CLAHE contrast enhancement)
- Machine Learning: Scikit-learn (K-Means clustering for cell classification)
- Visualization: Matplotlib (result visualization and image saving)
- Environment: Python 3.8+ / Jupyter Notebook (fully compatible with both .py and .ipynb formats)

## How to Run
### Step 1: Install Dependencies
```bash
pip install opencv-python numpy matplotlib scikit-learn