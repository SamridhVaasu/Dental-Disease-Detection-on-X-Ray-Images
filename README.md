# Dental-Disease-Detection-on-X-Ray-Images

![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 📌 Overview
This project presents an advanced **dental disease detection on X-Ray images** powered by **YOLO (You Only Look Once)**, integrated into a **Streamlit-based** web interface. The model efficiently processes **dental X-ray images** to identify and classify common dental conditions, including:

- 🦷 **Dental caries (cavities)**
- 🦷 **Restorative fillings**
- 🦷 **Impacted teeth**
- 🦷 **Dental implants**

## 🏗 Model Details
### 🔬 Architecture and Training Parameters
- **Model**: YOLO11n (nano version)
- **Input Resolution**: 640×640 pixels
- **Target Classes**: 4 (Cavities, Fillings, Impacted Tooth, Implant)
- **Hyperparameters**:
  - **Epochs**: 5
  - **Batch Size**: 16
  - **Optimizer**: AdamW (lr=0.00125, momentum=0.9)
  - **Dataset Composition**: 753 training images, 215 validation images

### 📊 Performance Metrics (mAP50)
| Condition          | Detection Accuracy (mAP50) |
|-------------------|--------------------------|
| **Overall**       | 0.603                    |
| **Implants**      | 0.916                    |
| **Fillings**      | 0.827                    |
| **Impacted Teeth** | 0.644                   |
| **Cavities**      | 0.0246                   |

## 🌟 Web Application Features
### 🔍 Image Processing and Analysis
- Supports **PNG, JPG, and JPEG** image formats.
- Real-time inference on **dental X-ray images**.
- Side-by-side display of **original and processed images**.

### 📌 Condition-Specific Analysis
- **Dedicated visualization tabs** for each detected pathology.
- **Confidence scores** for model predictions.
- Multi-perspective visualization:
  - 🖼 **Annotated full image with detection boxes**
  - 🔍 **Cropped region of interest (ROI)**
  - 🎯 **Focused view of marked ROI**

### 📈 Statistical and Graphical Representations
- 📊 **Distribution of detection confidence scores**.
- 🥧 **Class distribution pie charts**.
- 📉 **Interactive visualizations powered by Plotly**.

## ⚡ Installation & Setup
### 🖥 System Requirements
- **Python 3.9+**
- **Streamlit**
- **Ultralytics YOLO**
- **OpenCV**
- **Pillow**
- **Plotly**
- **NumPy**
- **Pandas**
- **Streamlit-lottie**

### 🔧 Installation Steps
1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage Guide
1. **Run the application:**
   ```bash
   streamlit run app.py
   ```
2. **Upload a dental X-ray image** via the web interface.
3. **Review the analysis**, including:
   - 🦷 Bounding-box annotations for detected conditions.
   - 📊 Model confidence scores per detection.
   - 🔎 Detailed breakdown of each pathology.
   - 📈 Statistical summaries and visual analytics.

## 🎯 Model Training
The model was trained using the **Ultralytics YOLO framework** with the command:
```bash
yolo detect train data=datasets/Dental-X-ray/data.yaml model=yolo11n.pt epochs=5 imgsz=640
```

## ⚠️ Limitations
- 🚫 **Cavity detection performance requires improvement** (mAP50: 0.0246).
- 📷 **Currently supports only X-ray image analysis**.
- 🕒 **Inference time depends on image resolution and system capabilities**.

## 🚀 Future Enhancements
✅ **Enhancing cavity detection through additional data augmentation**  
✅ **Expanding compatibility to 3D dental imaging**  
✅ **Integrating AI-driven treatment recommendations**  
✅ **Developing real-time video processing capabilities**  
✅ **Adding multilingual support for broader accessibility**  

## 🤝 Contributing
We welcome contributions! Please **fork the repository** and submit a **Pull Request** with any improvements or feature additions.

---
### 📧 Contact
For inquiries or collaborations, contact us via email: [samridh@iiotengineers.com](mailto:your-email@example.com)

