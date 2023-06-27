##  A brief Description of the Models we used
##

### Logistic Regression

Logistic regression is a popular classification model used to predict the probability of a binary outcome. It assumes a linear relationship between the input variables and the log-odds of the output variable. By applying a logistic function to the linear combination of input features, logistic regression maps the continuous input space to a probability range between 0 and 1. It is widely used due to its simplicity, interpretability, and efficiency for large datasets.
###

### Support Vector Machines

Support Vector Machines are powerful supervised learning models used for both classification and regression tasks. SVM aims to find an optimal hyperplane that separates the input data into different classes. It selects the hyperplane with the maximum margin, which is the largest distance between the closest points of different classes. SVM can handle both linearly separable and non-linearly separable data by using kernel functions that transform the data into a higher-dimensional feature space.
###

### Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem and the assumption of independence between features. It assumes that all features are conditionally independent given the class label. Despite this "naive" assumption, Naive Bayes often performs well and is computationally efficient. It calculates the probabilities of different classes for a given set of features and predicts the class with the highest probability.
###

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It creates a "forest" of decision trees by training each tree on a different subset of the training data and using a random subset of features at each split. During prediction, each tree in the forest independently produces a class prediction, and the final prediction is determined by majority voting or averaging. Random Forests are known for their robustness, scalability, and ability to handle high-dimensional data.
###

### K-Nearest Neighbours (KNN)

K-Nearest Neighbors is a non-parametric classification algorithm that assigns a class label to an input sample based on the majority class of its k nearest neighbors in the feature space. It operates on the principle that similar instances tend to belong to the same class. KNN does not learn a model from the training data but instead stores the entire dataset to make predictions. The value of k determines the level of smoothness in the decision boundary: smaller values of k lead to more complex and potentially noisy decision boundaries, while larger values of k provide smoother boundaries. KNN is simple to understand and implement but can be computationally expensive for large datasets.
###

