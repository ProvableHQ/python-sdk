# -*- coding: utf-8 -*-
import os
import unittest
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from leotranspiler import LeoTranspiler


class TestLeoTranspiler(unittest.TestCase):
    def test_init(self):
        leo_transpiler = LeoTranspiler(None)
        self.assertEqual(leo_transpiler.model, None)
        self.assertEqual(leo_transpiler.validation_data, None)
        self.assertEqual(leo_transpiler.ouput_model_hash, None)
        self.assertEqual(leo_transpiler.transpilation_result, None)
        self.assertEqual(leo_transpiler.leo_program_stored, False)

    def test_init_tree_run(self):
        # Import necessary libraries
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset into a training and a test set
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Transpile
        lt = LeoTranspiler(clf, X_test)
        lt.to_leo(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])
        self.assertEqual(lc.active_input_count, 3)

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), "leotranspiler", "tests", "tree1"))

    def test_init_tree_run_model_parameters_as_inputs(self):
        # Import necessary libraries
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset into a training and a test set
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Transpile
        lt = LeoTranspiler(clf, X_test)
        lt.to_leo(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree1", True)
        self.assertEqual(lt.leo_program_stored, True)

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])

        lc = lt.run(X_test[1])

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), "leotranspiler", "tests", "tree1"))

    def test_init_tree_execute(self):
        # Import necessary libraries
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset into a training and a test set
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Transpile
        lt = LeoTranspiler(clf, X_test)
        lt.to_leo(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Execute and compare the Python prediction with the Leo prediction
        zkp = lt.execute(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(zkp.output_decimal[0]), python_prediction[0])

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), "leotranspiler", "tests", "tree1"))

    def download_and_extract_dataset(self, url, save_path, folder_path):
        """Download and extract dataset if it doesn't exist."""
        if not os.path.exists(save_path):
            print("Downloading the dataset...")  # noqa: T201
            response = requests.get(url)
            with open(save_path, "wb") as file:
                file.write(response.content)
            with ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(folder_path)
            print(f"Dataset downloaded and extracted in {folder_path}/")  # noqa: T201
        else:
            print(f"Dataset already exists in {save_path}")  # noqa: T201

    def test_run_credit(self):  # noqa: D103
        # File and folder specifications
        url = "https://archive.ics.uci.edu/static/public/144/"
        "statlog+german+credit+data.zip"
        folder_name = "tmp"
        zip_file_name = "statlog+german+credit+data.zip"
        folder_path = os.path.join(os.getcwd(), "leotranspiler", "tests", folder_name)
        path_to_save = os.path.join(folder_path, zip_file_name)

        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
        self.download_and_extract_dataset(url, path_to_save, folder_path)

        # Column definitions
        columns = [
            "existing_account",
            "month_duration",
            "credit_history",
            "purpose",
            "credit_amount",
            "saving_bonds",
            "employment_status",
            "installment_rate",
            "status_sex",
            "debts_status",
            "resident_since",
            "property",
            "age",
            "installment_plans",
            "housing_status",
            "credit_number",
            "job",
            "people_liability",
            "telephone",
            "foreign",
            "result",
        ]

        # Load data
        data_filepath = os.path.join(folder_path, "german.data")
        df = pd.read_csv(data_filepath, sep=" ", header=None, names=columns)

        # Data pre-processing
        dummy_columns = [
            "credit_history",
            "purpose",
            "status_sex",
            "debts_status",
            "property",
            "installment_plans",
            "housing_status",
            "foreign",
            "existing_account",
            "saving_bonds",
            "telephone",
            "job",
            "employment_status",
        ]
        df_numerical = pd.get_dummies(df, columns=dummy_columns, drop_first=True)

        # Splitting dataset into training and test sets
        X = df_numerical.drop("result", axis=1)
        y = df["result"].replace({1: 0, 2: 1})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )

        # Training the model
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)

        # Prediction and metrics
        y_pred = clf.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  # noqa: T201
        print(f"AUC score: {roc_auc_score(y_test, y_pred):.2f}")  # noqa: T201

        # Transpile the deceision tree into Leo code
        lt = LeoTranspiler(model=clf, validation_data=X_train)
        lt.to_leo(os.path.join(os.getcwd(), "leotranspiler", "tests"), "tree_credit")

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test.iloc[0])
        python_prediction = clf.predict([X_test.iloc[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])

        # test another input
        lc = lt.run(X_test.iloc[1])

        # remove the generated folder
        import shutil

        shutil.rmtree(
            os.path.join(os.getcwd(), "leotranspiler", "tests", "tree_credit")
        )


if __name__ == "__main__":
    unittest.main()
