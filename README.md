# Ineng-135-Group-20

This is the description to the whole project:
Our task is to build a tool to predict metals’ prices in the future’s market to help users to identify when should trade. Why do we want to create this? Because we found even though there are similar platforms available, most of them are only accessible to people who are professional in the area of economics and finance such as bankers and licensed investors, rather than to the public. While there are a lot of non professional users who trade metals for personal investments and they also need a proper tool to help them to have a general idea when to buy in or sell out to increase their profit. Therefore, our project provides a good platform not only to professionals but also to non professionals and assists them to fulfill their economic desires. In our project, we collected the crude data from Kaggle and three different mathematical models have been applied to our data, which are linear model, time series and PCA model.

## Description of the python file -- linear regression 
At the beginning of this project, we were considering whether metal prices are linearly increasing with time or not. To prove this thought, we decided to pick gold as a reference, so we collected the monthly prices of gold from 1950 to July 2020. We split the data by 90% for training and 10 % for testing.  After that, we used a linear regression model from sklearn to build a linear model. We used this model to predict the values from 1950 to 2020,  and we plotted the line of prediction with the scatter of actual values, we could easily observe that the price of gold doesn't have a strong linear association with time. This graph disproofs our thought, even though the overall price went up with time, but it was choppy in the progress. We can see the price exploded more quadratically at some moments for some reasons and had retracements after the explosions. After that observation, we calculated  the mean square errors for the training set and testing set, which are huge!!!


1. Collect Monthly data of gold price.
2. Split the data into testing set and trainning set.
3. Build a linear regression model using sklearn.
4. Predict values.
5. Plot the predicted value and actual value.
6. Calculate the mean square error. 

## Description of python file -- PCA model
Since we got the result from linear regression model which is biased, so we had to use another linear model to substitute the method of linear regression. Then we picked PCA model. PCA means principal component analysis, which is a linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. First we applied logarithm on the raw data, because based on our data type, a log-transforming distribution might be a clearly better description of the data than a normal distribution. Then we trained the data. We set 10-years as a time period to train the data. And then we applied PCA to the data to get covariance values and PCA loadings. We set n_component as 3 so we got three sets of covariance values. If a covariance computed in PCA is large, then the corresponding principal component is important in describing the underlying data dependencies. Based on the three sets of covariance, we used the first set because the values are large. PCA loadings are the coefficients of the linear combination of the original variables from which the principal components are constructed. We used PCA loadings to get a virtual basket of the market of metals, which is used to compare with trained data of eachtype of metal to get their difference. And based on these up and down differences, we could predict what time to buy in or sell out. In order to show the accuracy of PCA model, we also applied binary prediction. Based on the results, we could see that PCA got higher accuracy than just using trained data and getting mean of the trained data. Therefore, we could say PCA is reliable.

1. Collect raw data from Kaggle
2. Apply logarithm on the data because we're using PCA model
3. Set 10-years as a time period to train the data, which is used to predict the next 10 years
4. Set n components as 3 to get three sets of PCA covariance
5. Get PCA loadings to build the virtual basket of the market of metals
6. Compare the trained data of metal with the virtual basket we got to get the up and down differences
7. Based on the differences, we could predict what time to sell out or buy in
8. To show PCA has higher accuracy, we also applied binary prediction to compare with other methods, like just using original data or getting mean of the data

## Description of python file -- Time Series Model
Description content

1. bullet point
2. bullt point

## Description of R file -- Gold Price Fully Analysis
Description content

1. bullet point
2. bullt point
