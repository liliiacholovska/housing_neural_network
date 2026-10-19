import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_network(X_train, y_train, epochs):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=epochs,
        random_state=42,
        tol=0.0,
        n_iter_no_change=301,
        early_stopping=False
    )

    print("\nПочаток навчання нейронної мережі...")

    model.fit(X_train, y_train)

    print("Навчання завершено!")
    print("Виконано епох:", model.n_iter_)

    return model


def evaluate_network(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Результати оцінювання ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    print("\nПерші 10 прогнозів:")

    for actual, predicted in zip(
        y_test.iloc[:10],
        predictions[:10]
    ):
        print(
            f"Реальна вартість: {actual:.3f} | "
            f"Прогноз: {predicted:.3f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.title("Зміна помилки під час навчання")
    plt.xlabel("Епоха")
    plt.ylabel("Функція втрат")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )
    plt.xlabel("Реальна вартість")
    plt.ylabel("Прогнозована вартість")
    plt.title("Порівняння прогнозів із реальними значеннями")
    plt.grid(True)
    plt.show()

    return predictions