import pandas as pd
df = pd.read_csv("classData.csv")
x = df [[ "Ia", "Ib", "Ic", "Va", "Vb", "Vc"]] #input Voltages and Currents in the x axis
y= df[["G","C","B","A"]] #output of FAULTS in the y axis


#plot Ia vs IB (Current of phase A vs Current of phase B) to see if there is a relationship between the two
import  seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="Ia", y="Ib", hue="G", data=df, alpha=0.5, palette="Set1")
plt.xlabel("Ia (Current of phase A)")
plt.ylabel("Ib (Current of phase B)")
plt.title("Relationship between Ia and Ib")
plt.show()
#will show an ellipse with many loops graph with AC waveform behavior 

#now train the model using RandomForestClassifier and evaluate its performance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)#splitting data into train and test sets
model= RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

#adding more predictors such as the Real Power (P) and Apparent Power (S) which can be calculated from the voltage and current values 

import numpy as np
df["P_approx"]= df["Va"]*df["Ia"]+df["Vb"]*df["Ib"]+df["Vc"]*df["Ic"] #approximation of real power using voltage and current values
#find magnitude of all voltages and currents to use as predictors
df["V_mag"] = np.sqrt(df["Va"]**2 + df["Vb"]**2 + df["Vc"]**2)  #Total voltage magnitude
df["I_mag"] = np.sqrt(df["Ia"]**2 + df["Ib"]**2 + df["Ic"]**2)  #Total current magnitude
df["Z_approx"] = df["V_mag"]/df["I_mag"].replace(0,np.nan)# approximation of Impedance this is NOT the correct way to calculate impedance since we are not given phase angle or RMS values 
#^this is all an approximation of impedance and should not be used for real world applications but it can be used as a predictor in our model to see if it improves accuracy
df["Z_approx"]=df["Z_approx"].fillna(0) #replace any NaN values with 0 in the column not calculation
df["S_approx"]= df["V_mag"]*df["I_mag"] 


#retrain the model with the new predictors
x2 = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "P_approx", "V_mag", "I_mag", "Z_approx", "S_approx"]]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)

#predict Using RandomForest 
model2 = RandomForestClassifier(random_state=42)
model2.fit(x2_train, y_train)
predictions2 = model2.predict(x2_test)

#Gradient Boosting Model For Comparison
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

model_gb = MultiOutputClassifier(GradientBoostingClassifier())
model_gb.fit(x2_train, y_train)

preds_gb = model_gb.predict(x2_test)


#XGBoosting Model For Comparison
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

model_xgb = MultiOutputClassifier(
   XGBClassifier(eval_metric="logloss", verbosity=0)
)

model_xgb.fit(x2_train, y_train)

preds_xgb = model_xgb.predict(x2_test)


#ALL Results Comparison
print("\n=== Model Performance Comparison ===")
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 35)

print(f"{'Baseline (RF)':<25} {accuracy_score(y_test, predictions):>10.4f}")
print(f"{'Random Forest (With Features)':<25} {accuracy_score(y_test, predictions2):>10.4f}")
print(f"{'Gradient Boosting':<25} {accuracy_score(y_test, preds_gb):>10.4f}")
print(f"{'XGBoost':<25} {accuracy_score(y_test, preds_xgb):>10.4f}")
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model2, x2, y, cv=kf, scoring="accuracy")

print("Cross-validation scores:", scores)
print("Average cross-validation accuracy:", scores.mean())
importance = pd.Series(model2.feature_importances_, index=x2.columns)
print("\n=== Feature Importance (Random Forest) ===")
importance_sorted = importance.sort_values(ascending=False)

for feature, value in importance_sorted.items():
    print(f"{feature:<15}: {value:.4f}")

importance.sort_values().plot(kind="barh", figsize=(8,5))
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

#Feature importance showed that impedance and current magnitude were the most significant predictors for electrical faults
#voltage(Va, Vc) were not as important
#To get more accurate readings include RMS values voltage and current values instead of just peak values and also include phase angle to calculate real impedance instead of just using the approximation we used in this code
# Feature engineering significantly improved model performance (88.7% → 95.3%).
# Impedance proxy and current magnitude were the most important predictors.
# XGBoost achieved the highest accuracy among tested models (~95.9%).