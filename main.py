import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, confusion_matrix

#DATA LOADING & PRE-PROCESSING
def load_and_preprocess(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    drop_cols = ['Patient_ID', 'Random_Protein_Sequence', 'Random_Gene_Sequence', 
                 'Gene/Factor', 'Chromosome_Location', 'Function', 'Effect', 'Category']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
    le_sex = LabelEncoder()
    le_fam = LabelEncoder()
    le_motor = LabelEncoder()
    le_cog = LabelEncoder()
    le_gene = LabelEncoder()
    le_stage = LabelEncoder()
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
    df_clean['Family_History'] = le_fam.fit_transform(df_clean['Family_History'])
    df_clean['Motor_Symptoms'] = le_motor.fit_transform(df_clean['Motor_Symptoms']) # Target for Model 3
    df_clean['Cognitive_Decline'] = le_cog.fit_transform(df_clean['Cognitive_Decline'])
    df_clean['Gene_Mutation_Type'] = le_gene.fit_transform(df_clean['Gene_Mutation_Type'])
    df_clean['Disease_Stage'] = le_stage.fit_transform(df_clean['Disease_Stage']) # Target for Model 1
    print("Data Preprocessing Complete.")
    return df_clean, le_stage, le_motor
df, le_stage_encoder, le_motor_encoder = load_and_preprocess('Huntington_Disease_Dataset.csv')

#1) MODEL 1: Random Forest Classification (Predicting Disease Stage)
print("\n--- Model 1: Random Forest Classifier (Target: Disease Stage) ---")
X_cls = df.drop(['Disease_Stage'], axis=1)
y_cls = df['Disease_Stage']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_c, y_train_c)
y_pred_c = rf_model.predict(X_test_c)
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c, target_names=le_stage_encoder.classes_))


#2) MODEL 2: Linear Regression (Predicting Functional Capacity)
print("\n--- Model 2: Linear Regression (Target: Functional Capacity) ---")
features_reg = ['Age', 'Chorea_Score', 'Brain_Volume_Loss', 'HTT_CAG_Repeat_Length']
target_reg = 'Functional_Capacity'
X_reg = df[features_reg]
y_reg = df[target_reg]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
y_pred_r = lr_model.predict(X_test_r)
print(f"Mean Squared Error: {mean_squared_error(y_test_r, y_pred_r):.2f}")
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.2f}")


#3) MODEL 3: Support Vector Machine (Predicting Motor Symptoms Severity)
print("\n--- Model 3: SVM Classifier (Target: Motor Symptoms Severity) ---")
X_svm = df[['HTT_CAG_Repeat_Length', 'Brain_Volume_Loss', 'Age', 'Protein_Aggregation_Level']]
y_svm = df['Motor_Symptoms']
scaler = StandardScaler()
X_svm_scaled = scaler.fit_transform(X_svm)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_svm_scaled, y_svm, test_size=0.2, random_state=42)
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train_s, y_train_s)
y_pred_s = svm_model.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test_s, y_pred_s))
print(f"Target Classes: {le_motor_encoder.classes_}")


#4) MODEL 4: Gradient Boosting Regressor (Predicting Chorea Score)
print("\n--- Model 4: Gradient Boosting (Target: Chorea Score) ---")
X_gb = df[['HTT_CAG_Repeat_Length', 'Brain_Volume_Loss', 'Age', 'HTT_Gene_Expression_Level']]
y_gb = df['Chorea_Score']
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_g, y_train_g)
y_pred_g = gb_model.predict(X_test_g)
print(f"Mean Squared Error: {mean_squared_error(y_test_g, y_pred_g):.2f}")
print(f"R2 Score: {r2_score(y_test_g, y_pred_g):.2f}")


#5) MODEL 5: K-Means Clustering (Unsupervised Patient Grouping)
print("\n--- Model 5: K-Means Clustering (Unsupervised Analysis) ---")
cluster_features = df[['HTT_CAG_Repeat_Length', 'Brain_Volume_Loss', 'Protein_Aggregation_Level']]
scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(cluster_features)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clust_scaled)
df['Cluster_Group'] = clusters
print("Cluster Centers (Scaled):\n", kmeans.cluster_centers_)
print("\nDistribution of clusters:\n", df['Cluster_Group'].value_counts())
print("\nCross-tabulation of Clusters vs Actual Disease Stage:")
print(pd.crosstab(df['Disease_Stage'], df['Cluster_Group'], rownames=['Actual Stage'], colnames=['Cluster']))
