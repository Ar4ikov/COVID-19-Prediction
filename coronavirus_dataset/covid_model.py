# | Created by Ar4ikov
# | Время: 13.05.2020 - 23:47
from keras.activations import relu

from coronavirus_dataset.generate_dataset import CovidDataset

from statistics import stdev

import numpy as np

from keras.layers import Input, Dense, Concatenate, Dropout, Reshape, LSTM, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K


class CovidModel:
    def __init__(self):
        self.covid_dataset = CovidDataset()
        self.dataset = self.covid_dataset.generate_dataset()
        self.days_since_1_22 = self.covid_dataset.days_since_1_22()

    def train_data(self, data, x_countries, number_days_per_batch, number_days_to_predict):
        X, Y = [], []

        for x in range(len(self.days_since_1_22)):
            if x + number_days_per_batch + number_days_to_predict >= len(self.days_since_1_22):
                continue

            for country in x_countries:
                if data[country]["all_time"]["confirmed"][x] < 8:
                    continue

                x_cases = data[country]["daily"]["confirmed"][x:x + number_days_per_batch]
                x_recover = data[country]["daily"]["recoveries"][x:x + number_days_per_batch]
                x_death = data[country]["daily"]["deaths"][x:x + number_days_per_batch]
                x_mortality = data[country]["statistic"]["mortality_rate"][x:x + number_days_per_batch]
                x_recovery = data[country]["statistic"]["recovery_rate"][x:x + number_days_per_batch]

                x_std_cases = data[country]["statistic"]["stdev"]["confirmed"][x:x + number_days_per_batch]
                x_std_rec = data[country]["statistic"]["stdev"]["recoveries"][x:x + number_days_per_batch]
                x_std_deaths = data[country]["statistic"]["stdev"]["deaths"][x:x + number_days_per_batch]

                batch_x = [i for i in
                           zip(x_cases, x_recover, x_death, x_mortality, x_recovery, x_std_cases, x_std_rec, x_std_deaths)]

                y_cases = data[country]["daily"]["confirmed"][
                          x + number_days_per_batch:x + number_days_per_batch + number_days_to_predict]
                y_recover = data[country]["daily"]["recoveries"][
                            x + number_days_per_batch:x + number_days_per_batch + number_days_to_predict]
                y_death = data[country]["daily"]["deaths"][
                          x + number_days_per_batch:x + number_days_per_batch + number_days_to_predict]

                batch_y = [i for i in zip(y_cases, y_recover, y_death)]

                X.append(batch_x)
                Y.append(batch_y)

        return np.array(X), np.array(Y)

    def return_last_days(self, number_of_days, country):
        data = self.dataset[country]["daily"]
        statistic = self.dataset[country]["statistic"]

        x_cases = data["confirmed"][::-1][:number_of_days][::-1]
        x_recover = data["recoveries"][::-1][:number_of_days][::-1]
        x_death = data["deaths"][::-1][:number_of_days][::-1]
        x_mortality = statistic["mortality_rate"][::-1][:number_of_days][::-1]
        x_recovery = statistic["recovery_rate"][::-1][:number_of_days][::-1]
        x_std_cases = statistic["stdev"]["confirmed"][::-1][:number_of_days][::-1]
        x_std_rec = statistic["stdev"]["recoveries"][::-1][:number_of_days][::-1]
        x_std_deaths = statistic["stdev"]["deaths"][::-1][:number_of_days][::-1]

        output = [x for x in
                  zip(x_cases, x_recover, x_death, x_mortality, x_recovery, x_std_cases, x_std_rec, x_std_deaths)]

        return np.array([output])

    @staticmethod
    def zero_mask_value(x):
        return K.relu(x)

    def build_lstm_concatenate(self, input_shape, output_shape):
        input_layer = Input(input_shape[1:])

        x = LSTM(64, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)

        x = Dense(60)(x)

        layers = []
        for i in range(output_shape[1]):
            layers.append(Dense(output_shape[2], activation=relu)(x))

        x = Concatenate()(layers)

        output_layer = Reshape(output_shape[1:])(x)

        model = Model(input_layer, output_layer)
        model.compile(optimizer=RMSprop(), loss="mse")
        model.summary()

        return model

    @staticmethod
    def Y_to_X_1(x_history, Y):
        X_new = []

        x_history = x_history[0].tolist()

        for day in Y[0].tolist():
            output = day

            cases = day[0]
            if cases == 0:
                output.append(0), output.append(0)
                continue

            mortality_rate = day[2] / cases
            recovery_rate = day[1] / cases
            stdev_cases = stdev([x[0] for x in x_history])
            stdev_rec = stdev([x[1] for x in x_history])
            stdev_deaths = stdev([x[2] for x in x_history])

            output.append(mortality_rate), output.append(recovery_rate)
            output.append(stdev_cases), output.append(stdev_rec), output.append(stdev_deaths)

            X_new.append(output)

        return np.array([X_new])

    @staticmethod
    def Y_to_X(x_history, Y, K):
        X_new = []

        x_cases = (np.array(x_history["confirmed"]) / K).tolist()
        x_recovery = (np.array(x_history["recoveries"]) / K).tolist()
        x_deaths = (np.array(x_history["deaths"]) / K).tolist()

        x_cases.extend([x[0] for x in Y[0].tolist()])
        x_recovery.extend([x[1] for x in Y[0].tolist()])
        x_deaths.extend([x[2] for x in Y[0].tolist()])

        for idx, day in enumerate(Y[0].tolist()):
            output = day

            get_slice = lambda x: x[:-len(Y[0].tolist()) + idx + 1] if idx + 1 < len(Y[0].tolist()) else x

            cases = day[0]
            if cases == 0:
                output.append(0), output.append(0)
                continue

            mortality_rate = day[2] / cases
            recovery_rate = day[1] / cases
            stdev_cases = stdev(get_slice(x_cases))
            stdev_rec = stdev(get_slice(x_recovery))
            stdev_deaths = stdev(get_slice(x_deaths))

            output.append(mortality_rate), output.append(recovery_rate)
            output.append(stdev_cases), output.append(stdev_rec), output.append(stdev_deaths)

            X_new.append(output)

        x_history["confirmed"] = x_cases
        x_history["recoveries"] = x_recovery
        x_history["deaths"] = x_deaths

        return np.array([X_new]), x_history

    @staticmethod
    def concatenate_data(x, y):
        x, y = x[0].tolist(), y[0].tolist()

        [x.pop(0) for _ in range(len(y))]
        [x.append(i) for i in y]

        return np.array([x])

    @staticmethod
    def mul(arr):
        x = 1
        for i in arr:
            x *= i

        return x
