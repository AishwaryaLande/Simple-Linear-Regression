# Simple-Linear-Regression delivery_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

deli=pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Simple Linear Regression/delivery_time.csv")
deli.info()
deli.rename(columns={'Delivery Time':'DT'},inplace=True)
deli.rename(columns={'Sorting Time':'ST'},inplace=True)
deli.describe()
deli.shape
deli.info()
deli.head()

np.mean(deli)
np.median(deli.DT) 
np.median(deli.ST) 
#  Dispersion
np.var(deli)
np.std(deli)

skew(deli.ST)  # Skewness
skew(deli.DT) # Skewness
kurtosis(deli.ST) # Kurtosis
kurtosis(deli.DT)  # Kurtosis

x = np.array(deli.ST)
y = np.array(deli.DT)
# Histogram
plt.hist(deli ['ST'],color='blue')
plt.hist(deli ['DT'],color = 'red')
# Boxplot 
sns.boxplot(deli,orient='v')
sns.boxplot(deli ['ST'],orient = 'v',color='coral')
sns.boxplot(deli ['ST'],orient = 'v',color='yellow')

sns.pairplot(deli)
sns.countplot(deli ['ST'])
sns.countplot(deli ['DT'])
# Normal Q-Q plot
plt.plot(deli);plt.legend(['Delivery_time','Sorting_time']);

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)
# Normal Probability Distribution 
x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(y))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Sorting_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Delivery_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')
# Scatter plot
plt.scatter(x,y,label='Scatter Plot',color='blue',s=20);plt.xlabel('Sorting_Time');plt.ylabel('Delivery_Time');plt.title('Scatter Plot')

np.corrcoef(deli['ST'], deli['DT'])
deli.corr()
sns.heatmap(deli.corr(), annot=True)
#  Simple Linear Regression Model1
model = smf.ols('DT~ST',data= deli).fit()
model.summary()  
model.params

pred = model.predict(deli)
error = deli.DT-pred
sum(error) 
# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
np.corrcoef(x,y)  
np.sqrt(np.mean(error**2))  
# Simple Linear Regression Model2, Apply Log transformation to X- variable
model1 = smf.ols('DT~np.log(ST)',data= deli).fit()
model1.summary()
model1.params

pred1 = model1.predict(deli)
error1 = deli.DT-pred1
sum(error1) 
# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
#help(plt.plot)
np.corrcoef(np.log(x),y)
np.sqrt(np.mean(error1**2))  
# Simple Linear Regression Model2, Apply Log transformation on 'Y'
model2 = smf.ols('np.log(deli.DT)~ST',data= deli).fit()
model2.summary()
model2.params
pred2 = model2.predict(deli)
error2= deli.DT-np.exp(pred2)
sum(error2) 
# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y)) 
np.sqrt(np.mean(error2**2)) 
# highest R-Squared value is of log transformation - model2
resid_2 = pred2- deli.DT
student_resid = model2.resid_pearson 
student_resid
plt.plot(model2.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred2,y=DT);plt.xlabel("Predicted");plt.ylabel("Actual")
