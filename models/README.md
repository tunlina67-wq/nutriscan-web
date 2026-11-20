# models/

Put trained model files here. The app detects them automatically on start.

| File | Comes from | Used for |
|---|---|---|
| `best.pt` | Colab notebook `NutriScan_YOLO_Training.ipynb` → `runs/nutriscan/weights/best.pt` | Cropping the nutrition table before OCR |
| `grade_rf.joblib` | `python train_classifier.py` | ML health grade (A–E) |

Without these files the app still runs: it reads the full photo and grades with the expert rules only.
