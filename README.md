# 📄 Resume Screening System

A Machine Learning and NLP-based **Resume Screening System** that analyzes a candidate's resume and predicts the most suitable career category using **TF-IDF** and a **LinearSVC** classification model.

The application accepts a resume in PDF format, extracts the text, preprocesses it, converts the text into numerical features using TF-IDF, and predicts the career category.

---

## 🚀 Project Overview

Recruiters may receive a large number of resumes for different types of jobs. Manually reviewing every resume can take significant time.

This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be used to automatically classify resumes into predefined career categories.

### Workflow

```text
Resume PDF
    ↓
Text Extraction
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
LinearSVC Model
    ↓
Predicted Career Category
```

---

## 🎯 Project Goal

The main goal of this project is to build a simple Resume Screening system that can automatically identify the career category that best matches the content of a resume.

---

## ✨ Features

* 📄 Upload resume in PDF format
* 🔍 Extract text from uploaded resume
* 🧹 NLP-based text preprocessing
* 🔢 TF-IDF feature extraction
* 🤖 Machine Learning classification
* 📊 Resume category prediction
* 🌐 Interactive Streamlit interface
* ⚡ Fast prediction using a saved ML model

---

## 🧠 Machine Learning

Two classification models were evaluated during development:

* Random Forest Classifier
* LinearSVC

The final model selected for the application is:

### LinearSVC

LinearSVC was selected based on the model comparison, particularly its **macro F1-score**, which is useful for evaluating performance across multiple resume categories.

### Model Comparison

| Model         | Accuracy | Macro F1 |
| ------------- | -------: | -------: |
| Random Forest |   72.43% |     0.65 |
| LinearSVC     |   72.03% |     0.68 |

Although Random Forest achieved slightly higher accuracy, LinearSVC achieved the higher macro F1-score. Since this is a multi-class resume classification problem with uneven category sizes, macro F1 was considered important when selecting the final model.

---

## 📚 Resume Categories

The model predicts one of the following categories:

* ACCOUNTANT
* ADVOCATE
* AGRICULTURE
* APPAREL
* ARTS
* AUTOMOBILE
* AVIATION
* BANKING
* BPO
* BUSINESS-DEVELOPMENT
* CHEF
* CONSTRUCTION
* CONSULTANT
* DESIGNER
* DIGITAL-MEDIA
* ENGINEERING
* FINANCE
* FITNESS
* HEALTHCARE
* HR
* INFORMATION-TECHNOLOGY
* PUBLIC-RELATIONS
* SALES
* TEACHER

---

## 🔧 NLP Preprocessing

The resume text goes through several preprocessing steps before being passed to the machine learning model.

The preprocessing includes:

1. Converting text to lowercase
2. Expanding contractions
3. Removing URLs
4. Removing HTML
5. Converting emojis into text
6. Removing punctuation
7. Tokenization
8. Removing English stopwords
9. Snowball stemming
10. Joining the processed tokens

This helps reduce unnecessary variations in the resume text before feature extraction.

---

## 🔢 TF-IDF

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert resume text into numerical features.

The vectorizer uses:

```text
ngram_range = (1, 2)
max_features = 10000
sublinear_tf = True
```

This allows the model to learn from both:

* Individual words (unigrams)
* Two-word combinations (bigrams)

For example:

```text
machine
learning
machine learning
```

can all contribute useful information to the classification model.

---

## 📊 Dataset

The project uses a resume dataset containing resume text and career categories.

The dataset contains approximately:

* **2,484 resumes**
* **24 career categories**

The dataset includes fields such as:

| Feature     | Description                   |
| ----------- | ----------------------------- |
| ID          | Resume identifier             |
| Resume_str  | Resume text                   |
| Resume_html | HTML representation of resume |
| Category    | Career category               |

The `Resume_str` column is used as the primary text source for machine learning.

`Resume_html` is not required for the initial NLP classification pipeline.

---

## 🖥️ Streamlit Application

The project includes a Streamlit web application.

### User Flow

```text
Open Application
      ↓
Upload Resume PDF
      ↓
Click "Analysis"
      ↓
Extract Resume Text
      ↓
Preprocess Text
      ↓
TF-IDF Transformation
      ↓
LinearSVC Prediction
      ↓
Display Career Category
```

Example output:

```text
According to the LinearSVC model,
This Resume is best suited for the 'HR' field.
```

---

## 📁 Project Structure

```text
Resume-Screening-System/
│
├── data_set/
│   └── Resume.csv
│
├── Models/
│   ├── Model.pkl
│   └── Vector.pkl
│
├── cv_xtractor/
│   └── extract_entities.py
│
├── Data_preprocessing.py
├── main.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* LinearSVC
* TF-IDF

### NLP

* NLTK
* Contractions
* Emoji processing
* Text preprocessing

### PDF Processing

* PDF text extraction

### Web Application

* Streamlit

### Model Serialization

* Pickle

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AliRaza265/Resume-Screening-System.git
```

Navigate into the project:

```bash
cd Resume-Screening-System
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment on Windows:

```bash
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a PDF resume and click the **Analysis** button to receive the predicted career category.

---

## 📈 Model Performance

The final model was evaluated on a held-out test set.

### LinearSVC

**Accuracy:** approximately **72.03%**

**Macro F1:** approximately **0.68**

**Weighted F1:** approximately **0.71**

The performance demonstrates that resume text contains useful information for predicting broad career categories.

However, some categories contain significantly fewer resumes than others, which can make prediction difficult for smaller classes.

---

## ⚠️ Limitations

This project is a **Resume Screening V1** system and has some limitations.

* It predicts a broad career category rather than a specific job position.
* It does not compare a resume against a specific job description.
* Performance can vary between career categories.
* Smaller categories have fewer training examples.
* Resume formatting and extraction quality can affect predictions.
* A predicted category should not be treated as a final hiring decision.

---

## 🔮 Future Improvements

Possible future versions can include:

### Version 2

**Resume + Job Description Matching**

```text
Resume
   +
Job Description
       ↓
Text Similarity
       ↓
Match Score
```

### Additional Features

* Resume-job matching score
* Skill extraction
* Education extraction
* Experience extraction
* Candidate ranking
* Missing skill detection
* Job recommendation
* Multiple resume comparison
* Improved NLP models
* Deep Learning-based text classification

---

## 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

* NLP preprocessing
* Text classification
* TF-IDF
* N-grams
* Train/Test splitting
* Multi-class classification
* Model comparison
* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Confusion Matrix
* Model serialization
* PDF text extraction
* Streamlit deployment

---

## 👨‍💻 Author

**Ali Raza**

Machine Learning | Python | NLP | Software Engineering

---

## ⭐ Project Status

**Completed — Resume Screening System V1**

The current version focuses on predicting the most suitable career category from an uploaded resume using NLP and Machine Learning.
