# Cell Pathology Analysis for AI Oncology Assessment
This repository is developed to complete an assessment assignment for AI Oncology, which includes two core tasks for histopathological image analysis.

## Assignment Tasks Overview
### Task 1: Cell Clustering & Annotation
- **Requirement**: Perform unsupervised clustering on cells in histopathological images to distinguish normal/abnormal (malignant) cells, and submit the annotated image with abnormal cells highlighted.
- **Implementation**: Combined Watershed algorithm for nuclei segmentation and K-Means clustering for cell classification. Abnormal cells are marked with red contours in the result image.

### Task 2: Cell Counting (To be completed)
- **Requirement**: Count the number of different types of cells (normal/abnormal) and provide annotated statistics.
- **Status**: This task will be implemented in the final version of the project.

## How to Run
### For Python Script (.py)
```bash
pip install opencv-python numpy matplotlib scikit-learn
python cell_clustering.py


