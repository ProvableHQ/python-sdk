# -*- coding: utf-8 -*-
# noqa: D100
import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def download_and_extract_dataset(url, save_path, folder_path):
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


def main():  # noqa: D103
    # File and folder specifications
    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    folder_name = "tmp"
    zip_file_name = "statlog+german+credit+data.zip"
    folder_path = os.path.join(os.getcwd(), "leotranspiler", "examples", folder_name)
    path_to_save = os.path.join(folder_path, zip_file_name)

    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    download_and_extract_dataset(url, path_to_save, folder_path)

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

    # Plot the decision tree
    plt.figure(figsize=(15, 7.5))
    plot_tree(
        clf, filled=True, feature_names=list(X.columns), class_names=["Good", "Bad"]
    )
    plt.show()


if __name__ == "__main__":
    main()
