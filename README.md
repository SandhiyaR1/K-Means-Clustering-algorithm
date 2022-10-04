# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
import pandas module to use the built-in functions for 
### Step2
Read the csv file.
### Step3
Scatter plot the applicant income and loan amount.
### Step4
Obtain the Kmean clustring for 2 classes.
### Step5
Predict the cluster group of Applicant Income and Loanamount.

## Program:
```python 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("Downloads/clustering.csv")
print(data.head(2))
x1=data.loc[:,['ApplicantIncome','LoanAmount']]
print(x1.head(2))
x=x1.values
sns.scatterplot(x[ :,0],x[ :,1])
plt.xlabel('Income')
plt.ylabel('Lonan')
plt.show()
KMean=KMeans(n_clusters=4)
KMean.fit(x)
print("cluster centers: ",KMean.cluster_centers_)
print('labels: ',KMean.labels_)
predicted_cluster=KMean.predict([[9200,110]])
print('the cluster group for the applicantincome 9200 and loan amount 110 is',predicted_cluster)

```
## Output:
![output](/kmeans.png)

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.