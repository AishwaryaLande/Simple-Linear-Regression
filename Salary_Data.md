# Simple-Linear-Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

salary=pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Simple Linear Regression/Salary_Data.csv")
salary.info()
salary.describe()
salary.shape
salary.head(10)
 
np.mean(salary.YerExp) 
np.mean(salary.Salary) 
np.median(salary.Salary) 
np.median(salary.YerExp) 

# despersion
np.std(salary)
np.var(salary.YerExp)
np.std(salary.YerExp) 
np.var(salary.Salary) 
np.std(salary.Salary)

skew(salary) # skewnesss
kurtosis(salary)  # kurtosis

x=np.array(salary.Salary)
y=np.array(salary.YerExp)
# Histogram
plt.hist(salary.YerExp)
plt.hist(salary.Salary)
# Boxplot
plt.boxplot(salary ["YerExp"])
plt.boxplot(salary ["Salary"])
sns.pairplot(salary)
sns.countplot(x)
sns.countplot(y)
# Normal Q-Q plot
plt.plot(salary);plt.legend(['YerExp','Salary_hike']); plt.show()
st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)
#Normal Probability Distribution
x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Years_of_Experience');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')
x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color = 'blue');plt.xlim(np.min(y),np.max(y)) ;plt.xlabel('Salary_hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')
 # scatter plot
plt.scatter(x,y,label='Scatter_plot',color='r',s=40);plt.xlabel('Years_of_Experience');plt.ylabel('Saalry_hike');plt.title('Scatter Plot ');
np.corrcoef(x,y) 
salary.corr()
sns.heatmap(salary.corr(), annot=True) # with annot=True values are seen of specific realtion
# Simple Linear Regression Model
model = smf.ols('Salary~YerExp',data= salary).fit()
model.summary()  
model.params

pred = model.predict(salary)
error = salary.Salary-pred
sum(error) 
# Scatter plot between X and Y
plt.scatter(x,y,color='blue');plt.plot(x,pred,color='green');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
np.corrcoef(x,y) 
np.sqrt(np.mean(error**2))  
#Simple Linear Regression Model, Apply Log transformation to X- variable
model1 = smf.ols('Salary~np.log(YerExp)',data= salary).fit()
model1.summary()
model1.params

pred1 = model1.predict(salary)
error1 = salary.Salary-pred1
sum(error1) 
# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='blue');plt.plot(np.log(x),pred1,color='green');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
#help(plt.plot)
np.corrcoef(np.log(x),y) 
np.sqrt(np.mean(error1**2))  # RMSE
# Simple Linear Regression Model2 , Apply Log transformation on 'Y'
model2 = smf.ols('np.log(Salary)~YerExp',data= salary).fit()
model2.summary()
model2.params
pred2= model2.predict(salary)
error2 = salary.Salary-np.exp(pred2)
sum(error2) 
# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='blue');plt.plot(x,pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y))  # RMSE 
np.sqrt(np.mean(error2**2)) 
# highest R-Squared value got from log transformation-model1
# getting residuals of the entire data set
resid_2 = pred2-salary.Salary
student_resid = model1.resid_pearson 
student_resid
plt.plot(model1.resid_pearson,'o');plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred2,y=salary.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")
