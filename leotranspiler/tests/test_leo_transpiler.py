import unittest
import sklearn
import os

from leotranspiler.leotranspiler.leo_transpiler import LeoTranspiler

class TestLeoTranspiler(unittest.TestCase):
    
    def test_init(self):
        leo_transpiler = LeoTranspiler(None)
        self.assertEqual(leo_transpiler.model, None)
        self.assertEqual(leo_transpiler.validation_data, None)
        self.assertEqual(leo_transpiler.model_as_input, False)
        self.assertEqual(leo_transpiler.ouput_model_hash, None)
        self.assertEqual(leo_transpiler.transpilation_result, None)
        self.assertEqual(leo_transpiler.leo_program_stored, False)
    
    def test_init_tree1(self):
        # Import necessary libraries
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset into a training and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        lt = LeoTranspiler(clf, X_test)
        lt.store_leo_program(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree1")

if __name__ == '__main__':
    unittest.main()