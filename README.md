# NashvilleHousing

In this project, the goal is to work on a dataset that has details of Nashville Housing and help a real estate company with its investments in the Nashville Area. The dataset to be used has information about the recent sales and contains details about the property as concerned with their Sale Price compared to value. Machine Learning techniques can be used to use given information, find the important factors affecting the target variable, and predict the target variable. In this project, we will use classification models like Logistic regression and tree-based models like decision tree, Random Forest, and Gradient Boost. We will be comparing the working efficiency and the results produced by each model to understand which works better for our investors.


Modeling

The dataset after all the cleaning of the variables and after performing one hot encoding has a total of 57 features and 1 target variable. The dataset had high categorical variables and therefore there is high dimensionality. Now, we will be performing Logistic regression and Tree Based Modeling and find out which works better and what features are significant to understand the value of the property. The dataset is split between train and test sets using sklearn’s train_test_split library. 70% data is used for training and 30% for testing purposes. We also found that our target variable classes are highly imbalance and can affect our model accuracy and therefore we performed an oversampling technique using SMOTE (Synthetic Minority Over-sampling Technique) which results into balanced classes.


Logistic Regression

We build a Logistic Regression model to predict if the Sale Price is ‘Under’ or ‘Over’ valued in order to help make a real estate company a huge investment into the growing Nashville area. We first get the summary of a logistic regression model. With the use of this summary we could find the significant variables and look at the P values we can say that “Sold as Vacant”, “Multiple
 
Parcels involved in Sale”, “Acreage”,“Land value”, “Building Value”, “Sale_Year”, “Sale_Month” are the most significant variables. After fitting the model, we see that the accuracy of the test is 63%. However, we get a very high false negative and false positive value.
Considering our problem statement we would want to reduce false negative values. We would not want a property to be called as over values when it is not as the real estate company wants to invest and make a profit. Moreover “Multiple Parcels Involved in Sale”, “Sale_Year”, and “Sale_Month” are negatively correlated while “Sold As Vacant” , “Acreage” are positively correlated with the target variable.

Decision Tree

We have built a decision tree model using “Gini index” criteria with an accuracy rate of 74%, and a precision value of 50% and recall of 45%. By comparison with the logistic regression model, the Decision Tree has achieved a higher accuracy rate, precision, and recall value. Moreover, we could find out the important features using the feature_importnace function and we see that Sale_Year is the most important feature followed by year_built, Sale_Month, Total_Value, and so on. On visualizing the decision tree, we can see that it splits the data on Sale_Year as we see that it is an important feature.

Random Forest

The concept behind Random Forest Model is using a large number of decision trees that operate as an ensemble which will help in achieving better predictive performance. The model has achieved an accuracy rate of 73% with a precision of 45% and a recall of 30%. The model is not performing better than the decision tree in this case. As Random Forest combines the answer of various decision trees we see that the feature importance of Random Forest is different and it includes more features as significant. With Random Forest there can be a possibility of overfitting.

Gradient Boosting
 
The gradient Boost Model helps to minimize the bias errors of the predictive model. We can see that the model has achieved an accuracy rate of about 78% with a precision rate of 63% and a recall value of 31%. This model has the highest accuracy rate of the above models. The False Negatives and False positive values are also less compared to any other model. But Gradient Boost Model takes more time to run than any other model.

Gradient Boosting
 
The gradient Boost Model helps to minimize the bias errors of the predictive model. We can see that the model has achieved an accuracy rate of about 78% with a precision rate of 63% and a recall value of 31%. This model has the highest accuracy rate of the above models. The False Negatives and False positive values are also less compared to any other model. But Gradient Boost Model takes more time to run than any other model.

Conclusion

In this project, we built four models to understand the pricing behavior for houses in the Nashville area. The features that affect the pricing behavior are - “Sold As Vacant”, “Multiple Parcels Involved in Sale”, “Finished Area”, “Sale_Year”, and “sale_Month” This means the pricing of the houses depends on the month of sale or the condition in which the house is being sold also the area and the value of the land. The best fit model would be a gradient boost as it gives a better accuracy score while handling bias errors.
 
If the real estate company has to make a decision based on the model results then the month should be considered while making decisions on property. Alos, the year of sale is an important factor.

The models were rightly able to identify the true positive values which is good for this business scenario as that will make the huge investment possible. However, the reason of getting high negative values could be because of high dimensionality of the dataset and the presence of multiple categorical variables having variety of categories in it.

Though Gradient Boost was a slower than other models it gave the most accurate predictions as it provides parallel tree boosting.

Reference:

●	Brownlee, J. (2020, June 11). Ordinal and One-Hot Encodings for Categorical Data. Machine Learning Mastery.
https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/#:~:te xt=Machine%20learning%20models%20require%20all
●	Justin, L. &. (2020, October 2). Logistic Regression Example in Python: Step-by-Step Guide. Just into Data. https://www.justintodata.com/logistic-regression-example-in-python/
●	scikit learn. (2009). 1.10. Decision Trees — scikit-learn 0.22 documentation. Scikit-Learn.org. https://scikit-learn.org/stable/modules/tree.html
●	Ray, S. (2015, December 4). Simple Methods to deal with Categorical Variables in Predictive Modeling. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-va riables-predictive-modeling/
