
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def create_datasets():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

    print("Навчальну та тестову вибірки створено!")

def analyze_data(df, name):
    print(f"\n--- {name} ---")
    print("Кількість записів:", len(df))
    print("Кількість стовпців:", len(df.columns))
    print("Пропущені значення:", df.isnull().sum().sum())
    print("Дублікати:", df.duplicated().sum())
    print("\nСтатистика:")
    print(df.describe())


def prepare_data(train, test):
    target = "MedHouseVal"

    train = train.drop_duplicates().copy()
    test = test.copy()

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test