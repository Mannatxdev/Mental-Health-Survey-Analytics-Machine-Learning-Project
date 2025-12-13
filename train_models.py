import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

df = pd.read_csv("../dataset/mental_health_survey.csv")

df["Stress_Score"] = df["Stress_Level"].astype(str).str[0].astype(int)
df["High_Stress"] = df["Stress_Score"].apply(lambda x: 1 if x >= 3 else 0)

features = [
    "Age_Group","Gender","Sleep_Hours","Screen_Time",
    "Free_Time_Activity","Exercise_Frequency",
    "Anxious_Exhausted_Frequency","Current_Emotional_State"
]

X = df[features]
y = df["High_Stress"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features)
])

stress_model = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

stress_model.fit(X, y)
joblib.dump(stress_model, "stress_model.pkl")

df["Help_Flag"] = df["Need_MH_Awareness_in_College"].map({"Yes":1,"No":0})

help_model = Pipeline([
    ("prep", preprocess),
    ("model", DecisionTreeClassifier(max_depth=5))
])

help_model.fit(X, df["Help_Flag"])
joblib.dump(help_model, "help_model.pkl")

sleep_map = {"Less than 6 hours":1,"6–8 hours":2,"More than 8 hours":3}
anxiety_map = {"Never":1,"Rarely":2,"Sometimes":3,"Often":4,"Always":5}

df["Sleep_Score"] = df["Sleep_Hours"].map(sleep_map)
df["Anxiety_Score"] = df["Anxious_Exhausted_Frequency"].map(anxiety_map)

cluster_data = df[["Stress_Score","Sleep_Score","Anxiety_Score"]].dropna()

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled)

joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models trained successfully")
