import numpy as np
import pandas as pd
data=pd.read_csv(r'C:\Users\omarn\OneDrive\سطح المكتب\asss\student_scores.csv')
x = data['Hours'].values
y = data['Scores'].values
theta_0 = 0
theta_1 = 0
alpha = 0.01
num_iterations = 100
sse_values=[]#theta_1*x+theta_0 = y  
for i in range(num_iterations):
    y_pred=theta_0 + theta_1 * x
    d_theta_0=-2 * np.sum(y-y_pred)
    d_theta_1=-2 * np.sum((y-y_pred)*x)
    theta_0 -= alpha*d_theta_0
    theta_1 -= alpha*d_theta_1
    mse=np.mean((y-y_pred)**2)
    sse_values.append(mse)
    if (i+1) %100 ==0:
        print(f"Iterations {i+1},MSE: {mse}")
print(f"Final parameters: theta_0={theta_0}, theta_1={theta_1}")