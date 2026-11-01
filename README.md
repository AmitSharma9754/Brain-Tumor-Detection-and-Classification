#  Design and Development of a Deep Learning-Based Brain Tumor Detection and Classification System

##  Project Overview

The **Brain Tumor Detection and Classification System** is a deep learning-based application designed to analyze brain MRI images and classify them into four categories using a Convolutional Neural Network (CNN).

The system integrates a trained CNN model with a Streamlit web application to provide MRI image analysis, tumor classification, prediction confidence, Grad-CAM visualization, and model performance evaluation.

The application is developed for educational and research purposes to demonstrate the application of deep learning in medical image analysis.

---

##  Project Objectives

- Develop a CNN-based brain tumor detection and classification model.
- Classify brain MRI images into four categories.
- Provide an interactive web application for MRI image analysis.
- Visualize model predictions using class probability distributions.
- Implement Grad-CAM for model explainability.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

---

##  Tumor Classification Categories

The system classifies MRI images into the following four categories:

| Category | Description |
|---|---|
| Glioma | A tumor originating from glial cells |
| Meningioma | A tumor arising from the meninges |
| Pituitary | A tumor originating in the pituitary gland |
| No Tumor | MRI image classified as having no tumor |

---

##  Key Features

- **MRI Image Upload:** Upload brain MRI images in JPG or PNG format.
- **Tumor Classification:** Predicts one of four brain MRI categories.
- **Prediction Confidence:** Displays the model's confidence score.
- **Grad-CAM Visualization:** Highlights image regions influencing the model's prediction.
- **Data Visualization:** Displays class probability distributions.
- **Model Performance Evaluation:** Provides accuracy, precision, recall, F1-score, and confusion matrix.
- **Interactive Dashboard:** User-friendly Streamlit interface.
- **About Section:** Includes project information, developer details, usage instructions, and medical disclaimer.

---

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| CNN | MRI image classification |
| Streamlit | Web application development |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| OpenCV | Image processing |
| Pillow | MRI image loading and processing |
| Scikit-learn | Model evaluation metrics |

---

##  System Architecture

The system follows the workflow below:

```text
         Brain MRI Image
                |
                ▼
         Image Upload
                |
                ▼
      Image Preprocessing
                |
                ▼
       Trained CNN Model
                |
                ▼
      Feature Extraction
                |
                ▼
      Tumor Classification
                |
                ▼
     Prediction Confidence
                |
       ┌────────┴────────┐
       ▼                 ▼
 Class Probability    Grad-CAM
  Distribution      Visualization
       |
       ▼
  Model Evaluation
```

---

##  CNN Model Architecture

The model uses a Convolutional Neural Network architecture for image feature extraction and classification.

```text
Input MRI Image
      |
      ▼
Convolutional Layer
      |
      ▼
ReLU Activation
      |
      ▼
Max Pooling Layer
      |
      ▼
Convolutional Layer
      |
      ▼
ReLU Activation
      |
      ▼
Max Pooling Layer
      |
      ▼
Flatten Layer
      |
      ▼
Fully Connected Layer
      |
      ▼
Output Layer
      |
      ▼
4-Class Classification
```

---

##  Project Structure

```text
Brain-Tumor-Detection-and-Classification/
│
├── app.py
├── train_model.ipynb
├── model_utils.py
├── preprocess.py
│
├── model/
│   └── brain_tumor_cnn.keras
│
├── screenshots/
│   ├── 01-homepage.png
│   ├── 02-tumor-prediction.png
│   ├── 03-mri-image-upload.png
│   ├── 04-data-visualization.png
│   ├── 05-gradcam-visualization.png
│   ├── 06-about-and-disclaimer.png
│   └── 07-model-performance-metrics.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Model Performance Evaluation

The model is evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |

The actual model performance values are available in the application's **Accuracy** section.

---

##  Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/AmitSharma9754/Brain-Tumor-Detection-and-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Brain-Tumor-Detection-and-Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

##  Model Training

To train the CNN model using the training notebook:

1. Open `train_model.ipynb` in Jupyter Notebook or VS Code.
2. Configure the dataset paths.
3. Execute the notebook cells.
4. Train the model using the prepared MRI dataset.
5. Save the trained model in the `model/` directory.

The application uses the saved trained model for MRI image classification.

---

##  Dataset

The dataset contains brain MRI images organized into training and testing categories.

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

**Note:** The dataset is not included in this repository. Please configure the dataset paths before training the model.

---

##  Future Enhancements

- Improve classification performance through hyperparameter optimization.
- Implement transfer learning using pretrained CNN architectures.
- Enhance Grad-CAM visualization.
- Deploy the application on a cloud platform.
- Explore additional medical image datasets.
- Improve model generalization and robustness.

---

##  Disclaimer

This project is developed strictly for educational and research purposes.

The predictions generated by this system are not intended to replace professional medical advice, diagnosis, or treatment. The model has not been validated for clinical use.

Always consult a qualified healthcare professional for medical evaluation and decisions.

---

##  Project Developers

This project was collaboratively developed by:

### Amit Sharma
B.Tech – Artificial Intelligence  
Madhav Institute of Technology and Science (MITS), Gwalior, India

GitHub: (https://github.com/AmitSharma9754)

### Krishna Gupta
B.Tech – Artificial Intelligence  
Madhav Institute of Technology and Science (MITS), Gwalior, India

GitHub: (https://github.com/Krishna28Gupta)

---

##  Acknowledgements

We would like to acknowledge our college, faculty members, and the open-source community for their support and resources throughout the development of this project.

---

**If you find this project useful, consider giving the repository a ⭐ on GitHub!**
