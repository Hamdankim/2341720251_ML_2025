# Machine Learning 2025
**NAMA:** Hamdan Azizul Hakim
**NIM:** 2341720251  
**Mata Kuliah:** Machine Learning  
**Tahun:** 2025

## 📋 Deskripsi
Repository ini berisi kumpulan praktikum dan tugas untuk mata kuliah Machine Learning. Setiap folder merepresentasikan sesi pembelajaran (JS) yang mencakup berbagai topik dari dasar hingga lanjutan dalam machine learning.

## 📂 Struktur Repository

### JS01 - Persiapan Environment
- **Topik:** Setup environment dan instalasi library
- **Library:** PyPREP, Scipy, wandb, pyECG
- **File:** `P1_JS01.ipynb`, `TP_JS01.ipynb`

### JS02 - Exploratory Data Analysis (EDA)
- **Topik:** Analisis data Titanic
- **Teknik:** 
  - Data loading dan preprocessing
  - Visualisasi distribusi (histogram, boxplot, bar chart)
  - Analisis korelasi dengan heatmap
  - Scatter plot untuk exploratory analysis
- **Library:** pandas, numpy, matplotlib, seaborn
- **File:** `P1_JS02.ipynb`, `P2_JS02.ipynb`, `P3_JS02.ipynb`, `P4_JS02.ipynb`, `TP_JS02.ipynb`

### JS03 - Data Preprocessing & Feature Engineering
- **Topik:** Persiapan data untuk machine learning
- **Teknik:**
  - Handling missing values (mean, mode, custom filling)
  - Feature extraction (FamilySize, Title, AgeBin, CabinDeck, FarePerPerson)
  - Label encoding
  - Standardization dengan StandardScaler
  - Stratified train-test split
- **Dataset:** Titanic, Breast Cancer
- **File:** `P1_JS03.ipynb`, `P2_JS03.ipynb`, `P3_JS03.ipynb`, `P4_JS03.ipynb`, `TP_JS03.ipynb`

### JS04 - Clustering: K-Means
- **Topik:** Unsupervised learning dengan K-Means
- **Teknik:**
  - Implementasi K-Means clustering
  - Metode Elbow untuk menentukan k optimal
  - Evaluasi dengan SSE (Sum of Squared Errors)
  - Visualisasi cluster dan centroid
  - Metrics: Completeness, V-measure, Adjusted Rand Index, Silhouette Coefficient
- **Dataset:** Iris
- **File:** `P1_JS04.ipynb`, `P2_JS04.ipynb`, `P3_JS04.ipynb`, `TP_JS04.ipynb`

### JS05 - Clustering: DBSCAN & HDBSCAN
- **Topik:** Density-based clustering
- **Teknik:**
  - DBSCAN (Density-Based Spatial Clustering)
  - HDBSCAN (Hierarchical DBSCAN)
  - Handling noise dan outliers
  - Visualisasi core samples vs noise
  - Perbandingan performa terhadap scale data
- **Library:** sklearn.cluster, hdbscan
- **File:** `P1_JS05.ipynb`, `P2_JS05.ipynb`, `TP_JS05.ipynb`

### JS06 - Regresi Linear
- **Topik:** Supervised learning untuk prediksi kontinyu
- **Teknik:**
  - Simple Linear Regression
  - Multiple Linear Regression
  - Train-test split (70:30)
  - Model training dengan statsmodels
  - Residual analysis
  - Evaluasi dengan R² score
  - Visualisasi regression line
- **Dataset:** E-commerce customer data (`dataset.csv`)
- **Metrics:** R², MSE, Residual plots
- **File:** `P1_JS06.ipynb`, `P2_JS06.ipynb`, `TP_JS06.ipynb`

### JS07 - Approximate Nearest Neighbors (ANN)
- **Topik:** Efficient similarity search
- **Algoritma:**
  - **Annoy** (Approximate Nearest Neighbors Oh Yeah)
  - **FAISS** (Facebook AI Similarity Search)
  - **HNSW** (Hierarchical Navigable Small World)
  - sklearn Nearest Neighbors (exact search)
- **Teknik:**
  - Build index dan query optimization
  - Benchmarking kecepatan (build time, query time)
  - Perbandingan akurasi vs speed trade-off
- **Use Case:** Large-scale vector similarity search
- **File:** `P1_JS07.ipynb` - `P6_JS07.ipynb`, `TP_JS07.ipynb`

### JS08 - Advanced Data Processing
- **Topik:** Data processing tingkat lanjut
- **Teknik:**
  - UCI ML Repository integration
  - Google Sheets integration
  - Label encoding untuk categorical data
  - Image preprocessing (resize, normalization)
  - MNIST dataset handling
- **Library:** ucimlrepo, tensorflow.keras.datasets, skimage
- **File:** `TP_JS08.ipynb`

### JS09 - Naive Bayes Classification
- **Topik:** Probabilistic classification
- **Algoritma:**
  - MultinomialNB untuk text classification
  - Gaussian Naive Bayes
- **Teknik:**
  - Text preprocessing dan vectorization
  - **CountVectorizer** (Bag of Words)
  - **TfidfVectorizer** (Term Frequency-Inverse Document Frequency)
  - Feature selection dengan Mutual Information
  - Evaluasi: accuracy, precision, recall, f1-score, confusion matrix
- **Dataset:** Text classification, spam detection
- **File:** `P1_JS09.ipynb`, `P2_JS09.ipynb`, `P3_JS09.ipynb`, `TP_JS09_1.ipynb`, `TP_JS09_2.ipynb`

### JS11 - Support Vector Machines (SVM)
- **Topik:** Supervised learning dengan margin maksimal
- **Kernel Types:**
  - Linear kernel
  - Polynomial kernel
  - RBF (Radial Basis Function) kernel
- **Teknik:**
  - SVC (Support Vector Classification)
  - SVR (Support Vector Regression)
  - Decision boundary visualization
  - Hyperparameter tuning (C, gamma)
  - Stratified train-test split variations (70:30, 80:20)
  - Feature extraction: histogram untuk image classification
- **Use Case:** 
  - Binary classification
  - Multi-class classification
  - Day/Night image classification
- **File:** `P1_JS11.ipynb` - `P5_JS11.ipynb`, `TP_JS11.ipynb`

### JS13 - Neural Networks Basics
- **Topik:** Deep learning fundamentals
- **Framework:** TensorFlow/Keras
- **Arsitektur:**
  - Feedforward Neural Networks (FNN)
  - Multi-layer perceptron
- **Aktivasi:** sigmoid, relu, softmax
- **Teknik:**
  - Backpropagation dari scratch (numpy)
  - Sequential API Keras
  - Dense layers
  - Optimizer: Adam, SGD
  - Loss functions: categorical_crossentropy, MSE
  - Model evaluation dan metrics
  - Learning rate tuning
  - Hyperparameter experiments (layer size, activation)
- **Dataset:** XOR problem, Iris, California Housing, MNIST
- **File:** `P1_JS13.ipynb`, `P2_JS13.ipynb`, `P3_JS13.ipynb`, `TP_JS13.ipynb`

### JS14 - Convolutional Neural Networks (CNN)
- **Topik:** Deep learning untuk computer vision
- **Arsitektur:**
  - Conv2D layers
  - MaxPooling2D
  - Flatten
  - Dropout untuk regularization
  - BatchNormalization
- **Teknik:**
  - Image augmentation (ImageDataGenerator)
  - Data normalization (pixel scaling)
  - Training dengan validation data
  - Model compilation dan fitting
  - Transfer learning concepts
- **Dataset:** 
  - Cats vs Dogs
  - CIFAR-10 (10 class image classification)
- **Metrics:** accuracy, loss, validation curves
- **File:** `P1_JS14.ipynb`, `P2_JS14.ipynb`, `TP_JS14.ipynb`

### UTS - Ujian Tengah Semester
- **Topik:** Clustering dan similarity search
- **Implementasi:**
  - K-Means clustering
  - DBSCAN clustering
  - Annoy untuk nearest neighbors
  - PCA untuk dimensionality reduction
- **Evaluasi:**
  - Silhouette Score
  - Davies-Bouldin Index
  - Cluster visualization
  - Noise detection
- **Dataset:** Heart disease dataset
- **File:** `UTS.ipynb`

## 🛠️ Tech Stack

### Core Libraries
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **Clustering:** sklearn.cluster, hdbscan
- **Similarity Search:** annoy, faiss, hnswlib
- **Image Processing:** PIL, opencv, skimage

### Specialized Tools
- **Text Processing:** CountVectorizer, TfidfVectorizer
- **Neural Networks:** tensorflow.keras
- **Statistical Modeling:** statsmodels
- **Data Sources:** ucimlrepo

## 📊 Topics Covered

### 1. **Unsupervised Learning**
   - K-Means Clustering
   - DBSCAN
   - HDBSCAN
   - PCA

### 2. **Supervised Learning - Regression**
   - Linear Regression (Simple & Multiple)
   - Support Vector Regression (SVR)

### 3. **Supervised Learning - Classification**
   - Naive Bayes (Multinomial, Gaussian)
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Neural Networks
   - Convolutional Neural Networks (CNN)

### 4. **Feature Engineering**
   - Missing value imputation
   - Feature extraction
   - Encoding (Label, One-Hot)
   - Normalization & Standardization
   - Text vectorization (BoW, TF-IDF)

### 5. **Model Evaluation**
   - Regression: R², MSE, MAE, Residual Analysis
   - Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   - Clustering: Silhouette Score, Davies-Bouldin Index, Elbow Method
   - Neural Networks: Loss curves, validation accuracy

### 6. **Advanced Topics**
   - Approximate Nearest Neighbors (ANN)
   - Hyperparameter Tuning
   - Model Comparison
   - Deep Learning Architectures
   - Image Classification

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow keras
pip install hdbscan annoy faiss-cpu hnswlib
pip install statsmodels ucimlrepo
```

### Usage
1. Clone repository
2. Navigate ke folder sesi yang diinginkan (e.g., `JS06/`)
3. Buka Jupyter Notebook
4. Jalankan cell secara berurutan

### Dataset Locations
- Lokal: File CSV disimpan di folder masing-masing (e.g., `JS06/dataset.csv`)
- Remote: Beberapa notebook menggunakan Google Colab dengan path `/content/`
- UCI Repository: Dataset diambil via `ucimlrepo.fetch_ucirepo()`

## 📝 Notes

### Path Handling
- **Local execution:** Gunakan relative path dari folder kerja
- **Google Colab:** Path dimulai dengan `/content/`
- Pastikan working directory sesuai saat menjalankan notebook

### Best Practices
- Selalu cek `data.head()`, `data.info()`, `data.describe()` sebelum modeling
- Lakukan train-test split untuk validasi
- Visualisasikan hasil sebelum dan sesudah modeling
- Dokumentasikan hasil eksperimen dan metrics

## 📈 Progress Tracking

| Sesi | Topik | Status |
|------|-------|--------|
| JS01 | Environment Setup | ✅ |
| JS02 | EDA | ✅ |
| JS03 | Preprocessing | ✅ |
| JS04 | K-Means | ✅ |
| JS05 | DBSCAN/HDBSCAN | ✅ |
| JS06 | Linear Regression | ✅ |
| JS07 | ANN Search | ✅ |
| JS08 | Advanced Processing | ✅ |
| JS09 | Naive Bayes | ✅ |
| JS11 | SVM | ✅ |
| JS13 | Neural Networks | ✅ |
| JS14 | CNN | ✅ |
| UTS | Clustering Project | ✅ |

## 👨‍💻 Author
**NAMA:** Hamdan Azizul Hakim <br>
**NIM:** 2341720251  
Machine Learning Course 2025

## 📄 License
Educational purposes - Machine Learning Course Materials
