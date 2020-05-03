# | Created by Ar4ikov
# | Время: 26.04.2020 - 15:53

import pandas as pd
from keras.metrics import accuracy
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from coronavirus_dataset.model import Coronavirus_prediction
from scipy.interpolate import interp1d
import numpy as np

stats = pd.read_csv("../owid-covid-data.csv")

stats_json = {}
columns = [x for x in stats.columns]

for item in stats.values:
    location = item[1]
    data = item[2]

    if location not in stats_json:
        if location not in ["World", "International"]:
            stats_json[location] = {data: {columns[i]: item[i] for i in range(len(columns)) if i not in [0, 1, 2]}}

    else:
        stats_json[location][data] = {columns[i]: item[i] for i in range(len(columns)) if i not in [0, 1, 2]}


def num_to_label(value):
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(value) - 1]


def get_statistic_by_country(name):
    return stats_json[name]


def get_not_null_date(name):
    country_stat = get_statistic_by_country(name)

    for idx, k in enumerate(country_stat.keys()):
        if country_stat[k]["total_cases"] > 250:
            return idx


def generate_statistic_by_country(name):
    country_stat = get_statistic_by_country(name)

    dates = [x for x in country_stat.keys()][get_not_null_date(name) - 1:]
    total_cases = [int(country_stat[x]["total_cases"]) for x in dates]

    dates = [num_to_label(x.split("-")[1]) + "-" + x.split("-")[2] for x in dates]

    fig, ax = plt.subplots()
    fig.set_size_inches(12.5, 8.5)

    # ax.set_yscale('log')
    ax.plot(dates, total_cases, color="red", lw=5, label=f"Total Cases in {name}")

    ax.legend(prop={'size': 20})

    plt.xticks(rotation=90)
    plt.show()


def show_train(history_):
    loss, acc = history_.get("loss"), history_.get("accuracy")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot([x for x in range(len(acc))], acc, color="orange", lw=2, label="Accuracy")
    ax2.plot([x for x in range(len(loss))], loss, color="red", lw=2, label="Loss")

    fig.set_size_inches(15, 8)

    ax1.legend(prop={"size": 15})
    ax2.legend(prop={"size": 15})

    plt.show()


# Dataset consts
K = 1000
TRAIN_DAYS_PAST = 30
DAYS_PER_ITERATION = 1
PREDICT_DAYS_AFTER = 120

metrics = ["total_cases", "new_cases", "total_deaths", "new_deaths"]
# output_metrics = ["total_cases", "new_cases"]

Coronavirus_model = Coronavirus_prediction(metrics=metrics, days=TRAIN_DAYS_PAST, days_after=DAYS_PER_ITERATION)

# # Total active cases
# Coronavirus_model.add_custom_metric(["total_cases", "total_deaths"], lambda x: x["total_cases"] - x["total_deaths"])
#
# # New active cases
# Coronavirus_model.add_custom_metric(["new_cases", "new_deaths"], lambda x: x["new_cases"] - x["new_deaths"])

X, Y = Coronavirus_model.train_data(stats_json)

X = X / K
Y = Y / K

print(Y.shape)
# print(Y[40])
# print(X[40])

model = Coronavirus_model.build_model()
model.summary()
history = model.fit(
    X, Y,
    epochs=300,
    batch_size=384,
    verbose=2,
)

# Show training results
show_train(history.history)

# Get prediction
X_test = np.array([Coronavirus_model.get_last_n_days(get_statistic_by_country("Ukraine")) / K])
print(X_test)

pool = np.array([])

# Predicting new N days after
for i in range(PREDICT_DAYS_AFTER):
    prediction = model.predict(X_test)
    # print(prediction)
    pool = np.append(pool, prediction[0])
    X_test = Coronavirus_model.compare_new_day(X_test, prediction)

pool = pool.reshape((PREDICT_DAYS_AFTER * DAYS_PER_ITERATION, Coronavirus_model.input_shape[1],))
print(pool)

total_cases_predicted = pool * K
# print(X_test)

# Show statistic
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_size_inches(20, 8)

ax1.bar([x + 1 for x in range(PREDICT_DAYS_AFTER * DAYS_PER_ITERATION)], [x[0] for x in total_cases_predicted], lw=2, color="green", label="Total cases predicted")
ax1.plot([x + 1 for x in range(PREDICT_DAYS_AFTER * DAYS_PER_ITERATION)], [x[0] for x in total_cases_predicted], lw=2, color="yellow")
ax1.legend(prop={"size": 15})
ax1.set_yscale('log')

ax2.bar([x + 1 for x in range(PREDICT_DAYS_AFTER * DAYS_PER_ITERATION)], [x[1] for x in total_cases_predicted], lw=2, color="orange", label="New cases predicted")
ax2.plot([x + 1 for x in range(PREDICT_DAYS_AFTER * DAYS_PER_ITERATION)], [x[1] for x in total_cases_predicted], lw=2, color="red")
ax2.legend(prop={"size": 15})
ax2.set_yscale('log')

plt.show()

# generate_statistic_by_country("Ukraine")
