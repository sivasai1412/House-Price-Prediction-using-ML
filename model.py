import numpy as np
import pandas as pd
import joblib
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
#load data
df = pd.read_csv("clean_data.csv")
df['bath'] = df['bath'].astype(int)
df['balcony'] = df['balcony'].astype(int)
df['total_sqft_int'] = df['total_sqft_int'].astype(int)
df['price_per_sqft'] = df['price_per_sqft'].astype(int)
# Split data
X= df.drop('price', axis=1)
y= df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51)
 
# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = joblib.load('RandomForestRegressionModel.pkl')
def predict_house_price(bath,balcony,total_sqft_int,bhk,price_per_sqft):
 
  x =np.zeros(len(X.columns))
  x[0]=bath
  x[1]=balcony
  x[2]=total_sqft_int
  x[3]=bhk
  x[4]=price_per_sqft

  x = sc.transform([x])[0]
 
  return model.predict([x])[0] 