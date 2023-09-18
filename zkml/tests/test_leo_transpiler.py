# -*- coding: utf-8 -*-
import os
import unittest
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from zkml import LeoTranspiler

library_name = "zkml"


class TestLeoTranspiler(unittest.TestCase):
    def test_init(self):
        leo_transpiler = LeoTranspiler(None)
        self.assertEqual(leo_transpiler.model, None)
        self.assertEqual(leo_transpiler.validation_data, None)
        self.assertEqual(leo_transpiler.output_model_hash, None)
        self.assertEqual(leo_transpiler.transpilation_result, None)
        self.assertEqual(leo_transpiler.leo_program_stored, False)

    def test_tree_iris_run(self):
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])
        self.assertEqual(lc.active_input_count, 3)

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree1"))

    def test_tree_iris_run_dataset(self):
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test[0:2])
        python_prediction = clf.predict(X_test[0:2])
        self.assertEqual(int(lc[0].output_decimal[0]), python_prediction[0])
        self.assertEqual(int(lc[1].output_decimal[0]), python_prediction[1])
        self.assertEqual(len(lc), 2)
        self.assertEqual(lc[0].active_input_count, 3)

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree1"))

    def test_tree_iris_run_model_parameters_as_inputs(self):
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree1", True)
        self.assertEqual(lt.leo_program_stored, True)

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])

        lc = lt.run(X_test[1])

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree1"))

    def test_tree_iris_execute(self):
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Execute and compare the Python prediction with the Leo prediction
        zkp = lt.execute(X_test[0])
        python_prediction = clf.predict([X_test[0]])
        self.assertEqual(int(zkp.output_decimal[0]), python_prediction[0])

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree1"))

    def test_tree_iris_execute_dataset(self):
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree1")
        self.assertEqual(lt.leo_program_stored, True)

        # Execute and compare the Python prediction with the Leo prediction
        zkp = lt.execute(X_test[0:2])
        python_prediction = clf.predict(X_test[0:2])
        self.assertEqual(int(zkp[0].output_decimal[0]), python_prediction[0])
        self.assertEqual(int(zkp[1].output_decimal[0]), python_prediction[1])
        self.assertEqual(len(zkp), 2)

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree1"))

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

    def test_tree_credit_run(self):  # noqa: D103
        # File and folder specifications
        url = "https://archive.ics.uci.edu/static/public/144/"
        url += "statlog+german+credit+data.zip"
        folder_name = "tmp"
        zip_file_name = "statlog+german+credit+data.zip"
        folder_path = os.path.join(os.getcwd(), library_name, "tests", folder_name)
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree_credit")

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test.iloc[0])
        python_prediction = clf.predict([X_test.iloc[0]])
        self.assertEqual(int(lc.output_decimal[0]), python_prediction[0])

        # test another input
        lc = lt.run(X_test.iloc[1])

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree_credit"))

    def test_tree_credit_run_dataset(self):  # noqa: D103
        # File and folder specifications
        url = "https://archive.ics.uci.edu/static/public/144/"
        url += "statlog+german+credit+data.zip"
        folder_name = "tmp"
        zip_file_name = "statlog+german+credit+data.zip"
        folder_path = os.path.join(os.getcwd(), library_name, "tests", folder_name)
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
        lt.to_leo(os.path.join(os.getcwd(), library_name, "tests"), "tree_credit")

        # Run and compare the Python prediction with the Leo prediction
        lc = lt.run(X_test.iloc[0:2])
        python_prediction = clf.predict(X_test.iloc[0:2])
        self.assertEqual(int(lc[0].output_decimal[0]), python_prediction[0])
        self.assertEqual(int(lc[1].output_decimal[0]), python_prediction[1])
        self.assertEqual(len(lc), 2)

        # remove the generated folder
        import shutil

        shutil.rmtree(os.path.join(os.getcwd(), library_name, "tests", "tree_credit"))

    def test_tree_mnist_run(self):  # noqa: D103
        import gzip
        import os
        import shutil

        import requests

        def download_and_extract_dataset(url, save_path, folder_path):
            """Download and extract dataset if it doesn't exist."""
            if not os.path.exists(save_path):
                print(f"Downloading {os.path.basename(save_path)}...")  # noqa: T201
                response = requests.get(url)
                with open(save_path, "wb") as file:
                    file.write(response.content)

                decompressed_file_name = os.path.splitext(os.path.basename(save_path))[
                    0
                ]
                decompressed_file_path = os.path.join(
                    folder_path, decompressed_file_name
                )

                with gzip.open(save_path, "rb") as f_in:
                    with open(decompressed_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(  # noqa: T201
                    f"{decompressed_file_name} downloaded and extracted."
                )  # noqa: T201
            else:
                print(f"{os.path.basename(save_path)} already exists.")  # noqa: T201

        # URLs and filenames
        file_info = [
            (
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "train-images-idx3-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ),
        ]

        folder_name = library_name + "/tests/tmp/mnist"
        folder_path = os.path.join(os.getcwd(), folder_name)

        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        # Download and extract each file
        for url, file_name in file_info:
            path_to_save = os.path.join(folder_path, file_name)
            download_and_extract_dataset(url, path_to_save, folder_path)

        import numpy as np

        def read_idx3_ubyte_image_file(filename):
            """Read IDX3-ubyte formatted image data."""
            with open(filename, "rb") as f:
                magic_num = int.from_bytes(f.read(4), byteorder="big")
                num_images = int.from_bytes(f.read(4), byteorder="big")
                num_rows = int.from_bytes(f.read(4), byteorder="big")
                num_cols = int.from_bytes(f.read(4), byteorder="big")

                if magic_num != 2051:
                    raise ValueError(f"Invalid magic number: {magic_num}")

                images = np.zeros((num_images, num_rows, num_cols), dtype=np.uint8)

                for i in range(num_images):
                    for r in range(num_rows):
                        for c in range(num_cols):
                            pixel = int.from_bytes(f.read(1), byteorder="big")
                            images[i, r, c] = pixel

            return images

        def read_idx1_ubyte_label_file(filename):
            """Read IDX1-ubyte formatted label data."""
            with open(filename, "rb") as f:
                magic_num = int.from_bytes(f.read(4), byteorder="big")
                num_labels = int.from_bytes(f.read(4), byteorder="big")

                if magic_num != 2049:
                    raise ValueError(f"Invalid magic number: {magic_num}")

                labels = np.zeros(num_labels, dtype=np.uint8)

                for i in range(num_labels):
                    labels[i] = int.from_bytes(f.read(1), byteorder="big")

            return labels

        folder_path = os.path.join(
            os.getcwd(), folder_name
        )  # Adjust this path to where you stored the files

        train_images = read_idx3_ubyte_image_file(
            os.path.join(folder_path, "train-images-idx3-ubyte")
        )
        train_labels = read_idx1_ubyte_label_file(
            os.path.join(folder_path, "train-labels-idx1-ubyte")
        )
        test_images = read_idx3_ubyte_image_file(
            os.path.join(folder_path, "t10k-images-idx3-ubyte")
        )
        test_labels = read_idx1_ubyte_label_file(
            os.path.join(folder_path, "t10k-labels-idx1-ubyte")
        )

        print(  # noqa: T201
            f"Shape of train_images: {train_images.shape}"
        )  # Should output "Shape of train_images: (60000, 28, 28)"
        print(  # noqa: T201
            f"Shape of train_labels: {train_labels.shape}"
        )  # Should output "Shape of train_labels: (60000,)"
        print(  # noqa: T201
            f"Shape of test_images: {test_images.shape}"
        )  # Should output "Shape of test_images: (10000, 28, 28)"
        print(  # noqa: T201
            f"Shape of test_labels: {test_labels.shape}"
        )  # Should output "Shape of test_labels: (10000,)"

        # Reshape the datasets from 3D to 2D
        train_images_2d = train_images.reshape(
            train_images.shape[0], -1
        )  # -1 infers the size from the remaining dimensions
        test_images_2d = test_images.reshape(test_images.shape[0], -1)

        # Create the classifier and fit it to the reshaped training data
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=10, random_state=0)
        clf.fit(train_images_2d, train_labels)

        import logging
        import os

        from zkml import LeoTranspiler

        # Set the logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Transpile the deceision tree into Leo code
        lt = LeoTranspiler(model=clf, validation_data=train_images_2d[0:200])
        leo_project_path = os.path.join(os.getcwd(), library_name + "/tests/tmp/mnist")
        leo_project_name = "tree_mnist_1"
        lt.to_leo(path=leo_project_path, project_name=leo_project_name)

        # Compute the accuracy of the Leo program and the Python program on the test set
        num_test_samples = len(test_images_2d)

        # let's limit the number of test stamples to 10 to make the computation faster
        num_test_samples = min(num_test_samples, 10)

        python_predictions = clf.predict(test_images_2d)

        leo_predictions = np.zeros(num_test_samples)
        for i in range(num_test_samples):
            one_dim_array = test_images_2d[i].ravel()  # noqa: F841
            inputs = lt.model_transpiler.generate_input(test_images_2d[i])  # noqa: F841

            leo_predictions[i] = lt.run(input_sample=test_images_2d[i]).output_decimal[
                0
            ]

            self.assertEqual(int(leo_predictions[i]), python_predictions[i])

    def test_tree_mnist_run_dataset(self):  # noqa: D103
        import gzip
        import os
        import shutil

        import requests

        def download_and_extract_dataset(url, save_path, folder_path):
            """Download and extract dataset if it doesn't exist."""
            if not os.path.exists(save_path):
                print(f"Downloading {os.path.basename(save_path)}...")  # noqa: T201
                response = requests.get(url)
                with open(save_path, "wb") as file:
                    file.write(response.content)

                decompressed_file_name = os.path.splitext(os.path.basename(save_path))[
                    0
                ]
                decompressed_file_path = os.path.join(
                    folder_path, decompressed_file_name
                )

                with gzip.open(save_path, "rb") as f_in:
                    with open(decompressed_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(  # noqa: T201
                    f"{decompressed_file_name} downloaded and extracted."
                )  # noqa: T201
            else:
                print(f"{os.path.basename(save_path)} already exists.")  # noqa: T201

        # URLs and filenames
        file_info = [
            (
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "train-images-idx3-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
            ),
            (
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ),
        ]

        folder_name = library_name + "/tests/tmp/mnist"
        folder_path = os.path.join(os.getcwd(), folder_name)

        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        # Download and extract each file
        for url, file_name in file_info:
            path_to_save = os.path.join(folder_path, file_name)
            download_and_extract_dataset(url, path_to_save, folder_path)

        import numpy as np

        def read_idx3_ubyte_image_file(filename):
            """Read IDX3-ubyte formatted image data."""
            with open(filename, "rb") as f:
                magic_num = int.from_bytes(f.read(4), byteorder="big")
                num_images = int.from_bytes(f.read(4), byteorder="big")
                num_rows = int.from_bytes(f.read(4), byteorder="big")
                num_cols = int.from_bytes(f.read(4), byteorder="big")

                if magic_num != 2051:
                    raise ValueError(f"Invalid magic number: {magic_num}")

                images = np.zeros((num_images, num_rows, num_cols), dtype=np.uint8)

                for i in range(num_images):
                    for r in range(num_rows):
                        for c in range(num_cols):
                            pixel = int.from_bytes(f.read(1), byteorder="big")
                            images[i, r, c] = pixel

            return images

        def read_idx1_ubyte_label_file(filename):
            """Read IDX1-ubyte formatted label data."""
            with open(filename, "rb") as f:
                magic_num = int.from_bytes(f.read(4), byteorder="big")
                num_labels = int.from_bytes(f.read(4), byteorder="big")

                if magic_num != 2049:
                    raise ValueError(f"Invalid magic number: {magic_num}")

                labels = np.zeros(num_labels, dtype=np.uint8)

                for i in range(num_labels):
                    labels[i] = int.from_bytes(f.read(1), byteorder="big")

            return labels

        folder_path = os.path.join(
            os.getcwd(), folder_name
        )  # Adjust this path to where you stored the files

        train_images = read_idx3_ubyte_image_file(
            os.path.join(folder_path, "train-images-idx3-ubyte")
        )
        train_labels = read_idx1_ubyte_label_file(
            os.path.join(folder_path, "train-labels-idx1-ubyte")
        )
        test_images = read_idx3_ubyte_image_file(
            os.path.join(folder_path, "t10k-images-idx3-ubyte")
        )
        test_labels = read_idx1_ubyte_label_file(
            os.path.join(folder_path, "t10k-labels-idx1-ubyte")
        )

        print(  # noqa: T201
            f"Shape of train_images: {train_images.shape}"
        )  # Should output "Shape of train_images: (60000, 28, 28)"
        print(  # noqa: T201
            f"Shape of train_labels: {train_labels.shape}"
        )  # Should output "Shape of train_labels: (60000,)"
        print(  # noqa: T201
            f"Shape of test_images: {test_images.shape}"
        )  # Should output "Shape of test_images: (10000, 28, 28)"
        print(  # noqa: T201
            f"Shape of test_labels: {test_labels.shape}"
        )  # Should output "Shape of test_labels: (10000,)"

        # Reshape the datasets from 3D to 2D
        train_images_2d = train_images.reshape(
            train_images.shape[0], -1
        )  # -1 infers the size from the remaining dimensions
        test_images_2d = test_images.reshape(test_images.shape[0], -1)

        # Create the classifier and fit it to the reshaped training data
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=10, random_state=0)
        clf.fit(train_images_2d, train_labels)

        import logging
        import os

        from zkml import LeoTranspiler

        # Set the logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Transpile the deceision tree into Leo code
        lt = LeoTranspiler(model=clf, validation_data=train_images_2d[0:200])
        leo_project_path = os.path.join(os.getcwd(), library_name + "/tests/tmp/mnist")
        leo_project_name = "tree_mnist_1"
        lt.to_leo(path=leo_project_path, project_name=leo_project_name)

        # Compute the accuracy of the Leo program and the Python program on the test set
        num_test_samples = len(test_images_2d)

        # let's limit the number of test stamples to 10 to make the computation faster
        num_test_samples = min(num_test_samples, 10)

        python_predictions = clf.predict(test_images_2d)

        leo_computations = lt.run(input_sample=test_images_2d[0:num_test_samples])

        leo_predictions = np.zeros(num_test_samples)
        for i in range(num_test_samples):
            leo_predictions[i] = leo_computations[i].output_decimal[0]

            self.assertEqual(int(leo_predictions[i]), python_predictions[i])


if __name__ == "__main__":
    unittest.main()
