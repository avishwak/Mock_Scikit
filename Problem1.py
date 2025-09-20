import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Telco-Customer-Churn.csv')

df.head()
df.info()
df.describe()

df.columns

X = df.iloc[:, 1:19] # Features dropping the first column (customerID)

label = LabelEncoder()
for col in cat_columns:
    X[col] = label.fit_transform(X[col])

df['Churn'] = label.fit_transform(df['Churn'])
y = df.iloc[:, 20]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression 
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy: .2f}")
cm = confusion_matrix(y_test, y_pred)
cm

# KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy: .2f}")

# SVC 
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy: .2f}")

# Random Forest 
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy: .2f}")