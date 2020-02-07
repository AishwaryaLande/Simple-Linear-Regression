# Simple-Linear-Regression calories_consumed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

calo= pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Simple Linear Regression/calories_consumed.csv ")
calo.info()
calo.describe()
calo.shape
calo.columns
calo.head()
calo.corr()

#calo.rename(columns={'WG (grams)':'__'},inplace =True)  # columns name changed
#calo.rename(columns={'CC':'____'},inplace=True)      
#Central Tendency
np.mean(calo.WG) 
np.mean(calo.CC) 
np.median(calo.WG) 
np.median(calo.CC) 
#Dispersion
np.std(calo)
np.var(calo)
skew(calo) # skewnesss
kurtosis(calo) # kurtosis
x=np.array(calo.CC)
y=np.array(calo.WG)
# Histogram
plt.hist(calo.CC)
plt.hist(calo.WG)
# Boxplot
plt.boxplot(calo ["WG"])
plt.boxplot(calo ["CC"])
#help(sns.boxplot)
#OR
sns.boxplot(calo,color='coral',orient='v')
sns.boxplot(calo ['CC'],orient='v',color='red') # orient = 'v' -> Vertival
sns.boxplot(calo ['WG'],orient='v',color='yellow')  # orient = 'h' ->horizontal
# Normal Q-Q plot
plt.plot(calo);plt.legend(['Calories Consumed','Weight gained (grams)']); plt.show()
st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)
#Normal Probability Distribution
x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Calories_Consumed');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')
x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color = 'green');plt.xlim(np.min(y),np.max(y)) ;plt.xlabel('Weight_gained(grams)');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')
# scatter plot
plt.scatter(x,y,label='Scatter_plot',color='r',s=40);plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot ');
 np.corrcoef(x,y) 
cal.corr()
sns.heatmap(calo.corr(), annot=True)
sns.pairplot(calo)
sns.countplot(x)
sns.countplot(y)
#simple Regression model
model=smf.ols("WG ~ CC",data= calo).fit()
model.params
model.summary()   #0.897
pred= model.predict(calo)
error= calo.WG- pred
sum(error) 
#np.mean(error)
# Scatter plot between 'x' and 'y'
plt.scatter(x,y,color='red',s=40);plt.plot(x,pred,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation Coefficientscalo
np.corrcoef(x,y) 
np.sqrt(np.mean(error**2)) 
## simple Regression model1 , Apply log transformation on x-variables
model1 = smf.ols('WG~np.log(CC)',data= calo).fit()
model1.summary()
model1.params
pred1 = model1.predict(calo)
error1 = calo.wt-pred1
sum(error1) 
# Scatter Plot between log(x) and y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('log(Calories_Consumed)');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation coefficient (r)
np.corrcoef(np.log(x),y)  
np.sqrt(np.mean(error1**2)) 
# simple Regression Model2 with log transformation - Y-variable
model2 = smf.ols('np.log(WG)~CC',data= calo).fit()
model2.summary()  
model2.params
pred2 = model2.predict(calo)
error2 = calo.WG-np.exp(pred2)
sum(error2)    # Sum of Errors should be Zero 
# Scatter Plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('log(Weight_gained(grams))');plt.title('Scatter Plot')
# Correlation Coefficient
np.corrcoef(x,np.log(y)) 
np.sqrt(np.mean(error2**2)) 
# highest R-Squared value is of log transformation - model2
# getting residuals of the entire data set
resid_2 = pred2-calo.WG
student_resid = model2.resid_pearson 
student_resid
plt.plot(model2.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred2,y=WG);plt.xlabel("Predicted");plt.ylabel("Actual")
