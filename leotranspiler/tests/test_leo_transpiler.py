import unittest
import os
from sklearn.tree import DecisionTreeClassifier

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

        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset into a training and a test set
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Transpile
        lt = LeoTranspiler(clf, X_test)
        lt.store_leo_program(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Prove and compare the Python prediction with the Leo prediction
        zkp = lt.prove(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(zkp.output[0]), python_prediction[0])


if __name__ == '__main__':
    unittest.main()