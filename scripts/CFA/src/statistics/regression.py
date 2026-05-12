import numpy as np


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X = self._add_intercept(X)
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        X = np.asarray(X)
        X = self._add_intercept(X)
        return X @ np.concatenate(([self.intercept], self.coefficients))

    def _add_intercept(self, X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def score(self, X, y):
        predictions = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        return 1 - (ss_residual / ss_total)


def main():
    import pandas as pd

    # Example usage
    data = pd.DataFrame(
        {"X1": [1, 2, 3, 4, 5], "X2": [2, 3, 4, 5, 6], "y": [3, 4, 2, 5, 6]}
    )

    X = data[["X1", "X2"]].values
    y = data["y"].values

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r_squared = model.score(X, y)

    print("Predictions:", predictions)
    print("R-squared:", r_squared)


if __name__ == "__main__":
    main()
