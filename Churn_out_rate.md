# Simple-Linear-Regression Churn_out_rate
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

emp= pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Simple Linear Regression/emp_data.csv")
emp.info()
emp.describe()
emp.shape
emp.rename(columns={'Salary_hike':'salary'},inplace=True)
emp.rename(columns={'Churn_out_rate':'cr'},inplace=True)
emp.describe()

np.mean(emp)
np.median(emp.salary) # 1675
np.median(emp.cr) # 71

# Measures of Dispersion
np.var(emp)
np.std(emp)


skew(emp.salary)    # Skewness
skew(emp.cr)      

kurtosis(emp.salary)  #Kurtosis
kurtosis(emp.cr)


x = np.array(emp.salary)
y = np.array(emp.cr)

# Normal Q-Q plot
plt.plot(emp.salary)
plt.plot(emp.cr)

plt.plot(emp);plt.legend(['salary','cr']);

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)

# Normal Probability Distribution plot 

x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Salary_Hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Churn_out_rate');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(emp['salary'],color='olive')

plt.hist(emp['cr'],color='skyblue')

# Boxplot 
sns.boxplot(emp,orient='v')
sns.boxplot(emp['salary'],orient = 'h',color='coral')
sns.boxplot(emp['ch_rate'],orient = 'h',color='skyblue')

sns.pairplot(emp)
sns.countplot(emp['salary'])
sns.countplot(emp['cr'])

# Scatter plot
plt.scatter(x,y,label='Scatter plot',color='coral',s=20);plt.xlabel('Salary_Hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot');
np.corrcoef(emp['salary'],emp['cr']) 
emp.corr()
sns.heatmap(emp.corr(),annot=True)
#Simple Linear Regression Model
model = smf.ols('cr~salary',data=emp).fit()
model.summary()  
model.params

pred = model.predict(emp)
error = emp.cr-pred
sum(error) 

# Scatter plot between X and Y
plt.scatter(x,y,color='coral');plt.plot(x,pred,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
np.corrcoef(x,y) 
np.sqrt(np.mean(error**2))

# Simple Linear Regression Model2, Apply Log transformation to X- variable
model1 = smf.ols('cr~np.log(salary)',data=emp).fit()
model1.summary() 
model1.params
pred1 = model1.predict(emp)
error1 = emp.cr-pred1
sum(error1)
# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='olive');plt.plot(np.log(x),pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
help(plt.plot)
np.corrcoef(np.log(x),y) 
# RMSE
np.sqrt(np.mean(error1**2)) 
# Simple Linear Regression Model3 , Apply Log transformation on 'Y'
model2 = smf.ols('np.log(cr)~salary',data=emp).fit()
model2.summary()  
model2.params
pred2 = model2.predict(emp)
error2 = emp.cr-np.exp(pred2)
sum(error2)

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='orange');plt.plot(x,pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y))

np.sqrt(np.mean(error2**2)) # RMSE
 #  model2 is having highest R-Squared value which is the log transformation - model2 
resid_2 = pred2-emp.cr
student_resid = model2.resid_pearson 
student_resid
plt.plot(model2.resid_pearson,'o');plt.axhline(y=0,color='skyblue');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred2,y=emp.ch_rate);plt.xlabel("Predicted");plt.ylabel("Actual")
plt.hist(model2.resid_pearson) # histogram for residual values 
