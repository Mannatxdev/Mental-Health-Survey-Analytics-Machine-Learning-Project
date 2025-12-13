Mental Health Survey Analytics – Machine Learning Project

Overview
This project focuses on analyzing the mental health status of students based on survey responses. 
Using both supervised and unsupervised Machine Learning models, the project predicts stress levels, 
help-seeking behavior, and identifies mental health risk groups for early intervention.

Objectives
1. Perform EDA to understand stress, anxiety & sleep patterns  
2. Predict high stress levels using Logistic Regression  
3. Predict help-seeking behavior using Decision Tree Classifier  
4. Cluster students into risk groups using K-Means Clustering  
5. Evaluate model performance using accuracy & silhouette score  

Tech Stack
- Python
- Pandas, NumPy, Scikit-Learn
- Matplotlib, Seaborn
- Jupyter Notebook

Machine Learning Models Used
| Model | Type | Purpose |
|------|------|---------|
| Logistic Regression | Classification | Predict high stress levels |
| Decision Tree Classifier | Classification | Predict help-seeking behavior |
| K-Means Clustering | Unsupervised Learning | Mental health risk segmentation |

Dataset Description
The dataset contains **1000 survey responses** with **16 attributes** including:
- Age group  
- Sleep hours  
- Anxiety frequency  
- Emotional state  
- Open communication behavior  
- Help-seeking behavior indicators  

Dataset: `mental_health_survey.csv`

Key Findings
- Poor sleep and high anxiety strongly contribute to high stress  
- Students hesitate to seek help even when mentally struggling  
- Clustering effectively identifies high-risk groups for support  


MindCare-AI-Frontend-Backend/
│
├── backend/
│ ├── app.py
│ ├── stress_model.pkl
│ ├── help_model.pkl
│ ├── kmeans.pkl
│ ├── scaler.pkl
│
├── frontend/
│ ├── index.html
│ ├── script.js
│ ├── style.css
│
├── dataset/
│ └── mental_health_survey.csv
---

**Developed by: Mannat Kumar**
