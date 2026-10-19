
import os
import pandas as pd

from data_loader import (
    create_datasets,
    analyze_data,
    prepare_data
)

from neural_network import (
    train_network,
    evaluate_network
)


def main():
    if (
      not os.path.exists("train.csv")
      or not os.path.exists("test.csv")
      or os.path.getsize("train.csv") == 0
      or os.path.getsize("test.csv") == 0
  ):
      create_datasets()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    analyze_data(train, "Навчальна вибірка")
    analyze_data(test, "Тестова вибірка")

    X_train, X_test, y_train, y_test = prepare_data(
        train,
        test
    )

    print("\nПідготовку даних завершено!")

    epochs = int(
        input("\nВведіть кількість епох навчання: ")
    )

    model = train_network(
        X_train,
        y_train,
        epochs
    )

    evaluate_network(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()