# | Created by Ar4ikov
# | Время: 02.05.2020 - 21:07
from typing import List, Optional

from keras.layers import Input, Dense, Reshape, LSTM, Flatten, Dropout, Embedding, SpatialDropout1D, Concatenate
from keras.activations import relu, sigmoid, tanh, softmax, linear
from keras.optimizers import RMSprop, Adam, Nadam, Adadelta, SGD, Adagrad, Adamax
from keras.models import Model, load_model
from keras.regularizers import l1, l2
from keras.metrics import accuracy
import numpy as np


class Coronavirus_prediction:
    def __init__(self, metrics: Optional[List[str]] = None, output_metrics: Optional[List[str]] = None, days=10, model_name=None, days_after=1):
        """
        Model for prediction a pick of Coronavirus pandemia
        Shape of keras model
        - [[total_cases, new_cases, total_deaths, new_deaths, new_cases_per_m, new_death_per_m] for x in days]
            or
        - [[x, x, x, x, x, x], [], [], [], [], [] ... ]
            or
        - [n, 6]

        :param input_shape: Array shape
        """

        self.output_metrics_state = False if output_metrics is None else True

        if output_metrics is None:
            output_metrics = metrics

        self.metrics: List[str] = metrics
        self.output_metrics: List[str] = output_metrics

        self.days = days
        self.days_after = days_after

        self.input_shape = (days, len(metrics), )
        self.output_shape = (days_after, len(output_metrics), )

        self.custom_metrics = []

        if model_name is None:
            model_name = "Coronavirus-2019"

        self.name = model_name

    @staticmethod
    def multiply(arr):
        x = 1
        for i in arr:
            x *= i

        return x

    @staticmethod
    def nan_to_zero(value):
        if isinstance(value, type(np.nan)):
            return 0.
        else:
            return value

    def recalculate_shape(self):
        self.input_shape = (self.days, len(self.metrics + self.custom_metrics), )

        if not self.output_metrics_state:
            self.output_shape = (self.days_after, len(self.output_metrics + self.custom_metrics), )

        return True

    def add_custom_metric(self, axis, func):
        self.custom_metrics.append({"axis": axis, "metric": func})
        self.recalculate_shape()

        return True

    def get_metrics(self, value):
        values = []
        for metric in value.items():
            k, v = metric

            if k in self.metrics:
                values.append(self.nan_to_zero(v))

        for c_metric in self.custom_metrics:
            values.append(self.nan_to_zero(c_metric["metric"](value)))

        return values

        # return [self.nan_to_zero(v) for k, v in value.items() if k in self.metrics]

    def get_output_metrics(self, value):
        values = []
        for metric in value.items():
            k, v = metric

            if k in self.output_metrics:
                values.append(self.nan_to_zero(v))

        for c_metric in self.custom_metrics:
            values.append(self.nan_to_zero(c_metric["metric"](value)))

        return values

    @staticmethod
    def get_not_null_date(data):
        country_stat = data

        for idx, k in enumerate(country_stat.keys()):
            if country_stat[k]["total_cases"] > 5:
                return idx

        return len(country_stat.keys())

    def get_last_n_days(self, data):
        return np.array([self.get_metrics(x) for x in data.values()][::-1][:self.days][::-1])

    @staticmethod
    def compare_new_day(x_input, prediction):
        x_input = x_input.tolist()

        [x_input[0].pop(0) for i in range(prediction.shape[1])]
        [x_input[0].append(x) for x in prediction[0]]

        return np.array(x_input)

    def get_day_pairs_per_country(self, country_stat):
        if len(country_stat.keys()) - self.get_not_null_date(country_stat) - 1 >= self.days + self.days_after:
            pairs = []

            dates = [x for x in country_stat.keys()]
            country_values = [self.get_metrics(x) for x in country_stat.values()]
            country_outs = [self.get_output_metrics(x) for x in country_stat.values()]

            for i in range(self.get_not_null_date(country_stat), len(dates)):
                if i + self.days + self.days_after <= len(dates):
                    pairs.append([
                        country_values[i:i + self.days],
                        country_outs[i + self.days: i + self.days + self.days_after]
                    ])
            return np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs])

        else:
            return np.array([]), np.array([])

    def train_data(self, data):
        X, Y = [], []

        labels = [x for x in data.keys()]
        for label in labels:
            _x, _y = self.get_day_pairs_per_country(data[label])

            X.extend(_x)
            Y.extend(_y)

        return np.array(X), np.array(Y)

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        x = Flatten()(input_layer)
        x = Dense(10, activation=relu)(x)

        # TODO: Посмотри в статью, посмотрина формулы, предугадывай параметры и экспоненциальный рост (!!!)
        x1, x2 = Dense(15, activation=relu)(x), Dense(15, activation=relu)(x)
        x = Concatenate()([x1, x2])

        x = Dense(self.multiply(self.output_shape), activation=relu)(x)

        output_layer = Reshape(self.output_shape)(x)

        model = Model(input_layer, output_layer, name=self.name)
        model.compile(optimizer=RMSprop(), loss="mean_squared_error", metrics=["accuracy"])

        return model
