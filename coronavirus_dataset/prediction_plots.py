# | Created by Ar4ikov
# | Время: 14.05.2020 - 00:07

from matplotlib import pyplot as plt
import matplotlib as mpl
from keras.models import load_model
import numpy as np
from scipy.signal import savgol_filter as savgol

from coronavirus_dataset.covid_model import CovidModel


def show_prediction_plot(data, predict_days, country_name, total_cases, last_day, model_name=None):
    fig, (new, total) = plt.subplots(2, 1)

    fig.set_size_inches(13, 16)

    cases_error = min([x[0] for x in data]) / 10
    recovery_error = min([x[1] for x in data]) / 10
    deaths_error = min([x[2] for x in data]) / 10

    width = 0.34

    cases_axis = [x for x in np.arange(predict_days)]
    recovery_axis = [x + width for x in cases_axis]
    deaths_axis = [x + width - 0.01 for x in recovery_axis]

    cases = savgol([x[0] for x in data], predict_days - 1, 4)

    print([x[0] for x in data])

    new.bar(cases_axis, [x[0] for x in data], yerr=cases_error,
            width=width, color="#c7011a", label="Прогноз новых случаев")
    new.bar(recovery_axis, [x[1] for x in data], yerr=recovery_error,
            width=width, color="#739c3e", label="Прогноз новых случаев излечения")
    new.bar(deaths_axis, [x[2] for x in data], yerr=deaths_error,
            width=width, color="gray", label="Прогнох новых смертей")

    new.plot(cases_axis, np.maximum(np.zeros(predict_days), cases),
             lw=4, color="#f04338", label="Савицкий-Гол фильтр (по динамике заражений)")

    total_error = min(total_cases) / 10

    total.bar(np.arange(predict_days), [x for x in total_cases],
              yerr=total_error, lw=2, color="#ff8f00", label="Прогноз всех случаев")

    new.set_title(f"Спрогнозированная Динамика COVID-19 в {country_name} (модель `{model_name}`)", fontsize=20)
    total.set_title(f"Суммарное количество случаев COVID-19 в {country_name} (модель `{model_name}`)", fontsize=20)

    new.set(xlabel=f"Дней с {last_day}", ylabel="# Случаев")
    total.set(xlabel=f"Дней с {last_day}", ylabel="# Случаев")

    mpl.rc('xtick', labelsize=20)
    mpl.rc('ytick', labelsize=20)

    new.legend(prop={"size": 12})
    new.legend(prop={"size": 12})

    plt.show()


covid = CovidModel()

covid_dataset = covid.covid_dataset
covid_dataset.filename = "covid-19-5-14-20.json"
covid_dataset.generate_dataset()

dataset = covid_dataset.dataset

covid_model_key = load_model("covid_model_5-14-20_key_1.hdf5")
covid_model_all = load_model("covid_model_5-14-20_all_1.hdf5")

dates = covid_dataset.dates

# Предсказываем 3 * 10 дней
ITERATIONS = 3
K = 1000


def predict(country):
    X_test_key = covid.return_last_days(20, country) / K
    X_test_all = covid.return_last_days(20, country) / K

    history_key = dataset[country]["daily"]
    history_all = dataset[country]["daily"]

    Total_cases_key = [dataset[country]["all_time"]["confirmed"][-1], ]
    Total_cases_all = [dataset[country]["all_time"]["confirmed"][-1], ]

    summary_prediction_key = []
    summary_prediction_all = []

    for i in range(ITERATIONS):
        prediction_key = covid_model_key.predict(X_test_key)
        prediction_all = covid_model_all.predict(X_test_all)

        summary_prediction_key.extend(prediction_key[0] * K)
        summary_prediction_all.extend(prediction_all[0] * K)

        Y_key, history_key = covid.Y_to_X(history_key, prediction_key, K)
        Y_all, history_all = covid.Y_to_X(history_all, prediction_all, K)

        X_test_key = covid.concatenate_data(X_test_key, Y_key)
        X_test_all = covid.concatenate_data(X_test_all, Y_all)

        zero_or_value = lambda x: x if x > 0 else 0

        for x in prediction_key[0] * K:
            Total_cases_key.append(float(Total_cases_key[-1]) + zero_or_value(float(x[0])))

        for x in prediction_all[0] * K:
            Total_cases_all.append(float(Total_cases_all[-1]) + zero_or_value(float(x[0])))

    show_prediction_plot(summary_prediction_key, ITERATIONS * 10,
                         country, Total_cases_key[1:], dates[-1], "KEY_COUNTRIES")
    plt.savefig(f"{country}_{dates[-1].replace('/', '-')}_key.png")

    show_prediction_plot(summary_prediction_all, ITERATIONS * 10,
                         country, Total_cases_all[1:], dates[-1], "ALL_COUNTRIES")
    plt.savefig(f"{country}_{dates[-1].replace('/', '-')}_all.png")


PREDICT_COUNTRIES = ["Russia", "US", "Ukraine", "Brazil", "Canada", "United Kingdom", "Peru", "Mexico"]


for country in PREDICT_COUNTRIES:
    predict(country)
