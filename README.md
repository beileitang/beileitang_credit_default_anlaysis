# Credit default analysising by using Support Vector Machine and Logistic Regression

-The goal of this project is to build a loan default forecasting model by using logistic regression and Support Vector Machines (SVM) with RBF kernel for Peer to Peer Lending (P2P). By using ROC curve and F score to compare the accuracy of models, thus the best model can be used to evaluate whether the loan should be approved or reject. 

### Result:
* From logistic regression, we found there is a positive affect on the probability of loan default, which is not consistent with our hypothesis as well as the practice. 
* Even we did the logistic regression with the annual income unit as dollar, and the coefficient shows extremely close to zero. 
* We will consider converting this variable to different units in the future work.
* However, the result of SVM with RBF kernel did not meet our expectation, it might because we only extracted fully paid and charged off data as target variable which is only 5839 and before we extracted the data it was 130,000, but mostly shows the loan is in processes. 
* So, the dataset might be too small to explain the reason, we could collect more data in the future. 


## Authors
Beilei Tang
