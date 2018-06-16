from sklearn.datasets import load_iris
from sklearn import tree
# Load dataset
iris = load_iris()
# Print labels in the iris dataset
print(list(iris.target_names))

# Create Decision Tree Model
classifier = tree.DecisionTreeClassifier()
# Build Decision Tree using it fit function which takes a set of examples and target labels
classifier = classifier.fit(iris.data, iris.target)
# Make a prediction
print(classifier.predict([[5.1, 3.5, 1.4, 1.5]]))
