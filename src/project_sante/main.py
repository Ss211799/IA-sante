import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {"age":[25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "cholesterol":[180, 190, 200, 210, 220, 230, 240, 250, 260, 270]}
df = pd.DataFrame(data)
X = df[["age"]]
y = df["cholesterol"]
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)  
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, predictions, color='red', label='Régression linéaire')
plt.xlabel('Âge')
plt.ylabel('Cholestérol')
plt.title('Relation entre l\'âge et le cholestérol')
plt.legend()
plt.show()
print("Coefficients de la régression linéaire :")
print(f"Intercept: {model.intercept_}")
print(f"Pente: {model.coef_[0]}")   