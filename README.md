# Customer-churn-predictor

A **data science and AI engineering** project that predicts which customers are likely to stop being customers using machine learning.  
Deployed via **Streamlit**, **Docker**, and **GitHub Actions** for a full end-to-end production workflow.


## Features

- **Data Cleaning & EDA:** Preprocess and explore customer data.  
- **Machine Learning Model:** Train and evaluate models to predict churn (Logistic Regression and Random Forest).  
- **Interactive Streamlit App:** Web interface for real-time churn predictions.  
- **Dockerized Deployment:** Containerized app for consistent environments.  
- **CI/CD with GitHub Actions:** Automate testing and deployment.


## Tech Stack

| Category | Tools |
|-----------|--------|
| **Language** | Python |
| **Libraries** | scikit-learn, pandas, numpy, matplotlib, seaborn |
| **Frameworks** | Streamlit|
| **DevOps** | Docker, GitHub Actions |
| **Version Control** | Git & GitHub |

## Live Demo
👉 **Try the app:** [Customer Churn Predictor](https://customer-churn-predictor1.streamlit.app)

## 📁 Project Structure

```bash
Customer-churn-predictor/
│
├── app/
│   └── app.py               # Streamlit web app
│
├── models/
│   ├── best_model.pkl       # Trained Random Forest model
│   └── scaler.pkl           # Feature scaler
│
├── notebooks/
│   └── churn_analysis.ipynb # Data exploration & model training
│
├── requirements.txt         # Dependencies
├── Dockerfile               # Container setup
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions workflow
│
└── README.md
```


## Run Locally

### Clone the repository
```bash
git clone https://github.com/sharonE-coder/Customer-churn-predictor.git
cd Customer-churn-predictor
```

### Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # For Linux/Mac
venv\Scripts\activate          # For Windows
```

### Install dependencies
```bash 
pip install -r requirements.txt
``` 

### Run the Streamlit app
```bash
cd app
streamlit run app.py
```

### 🐳 Run with Docker
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Then open 👉 http://localhost:8501

### CI/CD Workflow
Every push to the main branch triggers:


Docker Build – create production container


Auto Deployment – deploy to Streamlit Cloud


Author
Nomdorah Marcus
| AI Engineer | Data Scientist |
📍 Based in Nigeria
🔗 https://github.com/SharonE-coder · www.linkedin.com/in/nomdorah-marcus-438262213

⭐ If you found this project helpful, give it a star!
