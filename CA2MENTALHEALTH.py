# OBJECTIVE 1: Mental Health Pattern Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\manna\OneDrive\Desktop\INT234-CA2\mental_health_survey.csv")
print(df.head())
print(df.shape)
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Filling Missing Object Columns with "Unknown"
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Stress, Sleep & Anxiety basic plots
plt.figure(figsize=(10,4))
sns.countplot(data=df, x="Stress_Level")
plt.title("Stress Level Distribution")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(data=df, x="Sleep_Hours")
plt.title("Sleep Hours vs Students")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(data=df, x="Anxious_Exhausted_Frequency")
plt.title("Anxiety / Exhaust Frequency")
plt.show()


# OBJECTIVE 2: Logistic Regression - High Stress Prediction

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Convert stress level to numeric score
df["Stress_Score"] = df["Stress_Level"].astype(str).str[0].astype(int)

# Create Target Variable
df["High_Stress_Flag"] = df["Stress_Score"].apply(lambda x: 1 if x >= 3 else 0)

features = [
    "Age_Group", "Gender", "Sleep_Hours", "Screen_Time",
    "Free_Time_Activity", "Exercise_Frequency",
    "Anxious_Exhausted_Frequency", "Current_Emotional_State"
]

X = df[features]
y = df["High_Stress_Flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)]
)

log_model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - High Stress Prediction")
plt.show()



# OBJECTIVE 3: Decision Tree - Help Seeking Prediction

from sklearn.tree import DecisionTreeClassifier

# Encode Yes/No
df["Help_Seeking_Flag"] = df["Need_MH_Awareness_in_College"].map({"Yes": 1, "No": 0})

X2 = df[features]
y2 = df["Help_Seeking_Flag"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=42)

preprocess2 = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X2.columns)]
)

dt_model = Pipeline([
    ("preprocess", preprocess2),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

dt_model.fit(X_train2, y_train2)
y_pred2 = dt_model.predict(X_test2)

print("Accuracy:", accuracy_score(y_test2, y_pred2))
print("\nClassification Report:\n", classification_report(y_test2, y_pred2))

cm2 = confusion_matrix(y_test2, y_pred2)
ConfusionMatrixDisplay(cm2).plot()
plt.title("Confusion Matrix - Help Seeking Prediction")
plt.show()


# OBJECTIVE 4: K-Means Clustering - Mental Health Risk Groups

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Feature engineering for clustering
sleep_map = {"Less than 6 hours":1, "6–8 hours":2, "More than 8 hours":3}
anxiety_map = {"Never":1,"Rarely":2,"Sometimes":3,"Often":4,"Always":5}

df["Sleep_Score"] = df["Sleep_Hours"].map(sleep_map)
df["Anxiety_Score"] = df["Anxious_Exhausted_Frequency"].map(anxiety_map)

cluster_features = ["Stress_Score", "Sleep_Score", "Anxiety_Score"]
cluster_df = df[cluster_features].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
cluster_df["Cluster"] = clusters

df["Risk_Cluster"] = clusters

print(cluster_df["Cluster"].value_counts())
print("Silhouette Score:", silhouette_score(scaled_data, clusters))

# Visualize clusters
plt.scatter(scaled_data[:,0], scaled_data[:,1], c=clusters, cmap='viridis')
plt.xlabel("Stress Score (scaled)")
plt.ylabel("Sleep Score (scaled)")
plt.title("K-Means Mental Health Risk Clusters")
plt.show()


# OBJECTIVE 5: Performance Summary of Models

print("\n=== Final Summary ===")
print("Logistic Regression -> Stress Prediction Accuracy:", accuracy_score(y_test, y_pred))
print("Decision Tree -> Help Seeking Prediction Accuracy:", accuracy_score(y_test2, y_pred2))
print("K-Means -> Silhouette Score (Cluster Quality):", silhouette_score(scaled_data, clusters))

print("\nCluster Risk Groups:")
print(cluster_df.groupby("Cluster")[cluster_features].mean())
