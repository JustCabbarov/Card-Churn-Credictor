import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df= pd.read_csv("customers.csv")
df= df.drop("customer_id", axis=1)
X= df.drop("churned", axis=1)
y= df["churned"]
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

scaler= StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)

rf=RandomForestClassifier()
xgb= XGBClassifier()

rf.fit(x_train_scaled, y_train)
xgb.fit(x_train_scaled, y_train)

rf_score= rf.score(x_test_scaled, y_test)
xgb_score= xgb.score(x_test_scaled, y_test)

print(f"Random Forest Test Score: {rf_score:.4f}")
print(f"XGBoost Test Score: {xgb_score:.4f}")

y_pred_rf= rf.predict(x_test_scaled)
y_pred_xgb= xgb.predict(x_test_scaled)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(xgb, "models/xgboost_model.pkl")   
joblib.dump(scaler, "models/scaler.pkl")