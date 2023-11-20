from scipy.special import erfc
import numpy as np
import pandas as pd
from pysr import PySRRegressor


def emg(x, mu, sigma, lambda_):
    return (
            (lambda_ / 2) * np.exp(
        (lambda_ / 2) * (2 * mu + lambda_ * sigma ** 2 - 2 * x)
    ) * erfc(
        (mu + lambda_ * sigma ** 2 - x) / (np.sqrt(2) * sigma)
    )
    )


def generate(bound, n):
    xmin, xmax = bound
    return np.random.rand(n) * (xmax - xmin) + xmin


def get_x_y(n: int = 50):
    bounds = [(-3, 3), (-2, 2), (1, 5), (1, 2)]

    x = pd.DataFrame(
        np.stack([generate(bound, n) for bound in bounds]),
        index=['x', 'mu', 'sigma', 'lambda']
    ).T
    y = emg(x['x'], x['mu'], x['sigma'], x['lambda'])

    x = x.values
    y = y.values
    return x, y


def get_model():
    model = PySRRegressor(
        niterations=4000,  # < Increase me for better results
        binary_operators=["+", "*", "/"],
        populations=30,
        population_size=66,
        unary_operators=[
            "exp",
            "erfc",
            "square(x)=x^2",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"square": lambda x: x ** 2, },
        # ^ Define operator for SymPy as well
        loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )
    return model
