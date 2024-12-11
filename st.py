from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from supertree import SuperTree # <- import supertree :)

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Initialize supertree
super_tree = SuperTree(model, X, y, iris.feature_names, iris.target_names)

print(f"{iris.feature_names}, \n{iris.target_names}")

# show tree in your notebook
super_tree.save_html("SS")

# with open("data.html", "w") as file:
    # file.write(super_tree)