# | Created by Ar4ikov
# | Время: 13.05.2020 - 23:52

from os import path
from keras.callbacks import ModelCheckpoint
from coronavirus_dataset.covid_model import CovidModel
from matplotlib import pyplot as plt


def generate_checkpoint_name(day, postfix):
    iter = 0

    while True:
        name = f"covid_checkpoint_{day.replace('/', '-')}_{postfix}_{iter}.hdf5"

        if path.isfile(name):
            iter += 1
            continue

        return name


def generate_model_name(day, postfix):
    iter = 0

    while True:
        name = f"covid_model_{day.replace('/', '-')}_{postfix}_{iter}.hdf5"

        if path.isfile(name):
            iter += 1
            continue

        return name


def show_train_plot(history):
    fig, (loss, val_loss) = plt.subplots(1, 2)

    hval_loss, hloss = history.values()

    loss.plot([x + 1 for x in range(len(hloss))], hloss, lw=2, color="orange", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Index")

    val_loss.plot([x + 1 for x in range(len(hval_loss))], hval_loss, lw=2, color="orange", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Index")

    loss.set_title("Training Loss", fontsize=12)
    val_loss.set_title("Validation Loss", fontsize=12)

    fig.set_size_inches(13, 5)

    plt.show()

    return None


covid = CovidModel()

# covid.covid_dataset.filename = "covid-19-5-13-20.json"
# covid.covid_dataset.generate_dataset()

covid_dataset = covid.covid_dataset.dataset

KEY_COUNTIRES = covid.covid_dataset.KEY_COUNTRIES
COUNTRIES = [x for x in covid_dataset.keys()]  # covid.covid_dataset.countries
days_since_1_22 = covid.covid_dataset.days_since_1_22()
dates = covid.covid_dataset.dates

TIME_SERIES_DAYS = 20
PREDICT_DAYS = 10

# Коэффициент, на который делятся все значения в датасете для более точного обучения
K = 1000

# Train Dataset of key countries
X_train_key, Y_train_key = covid.train_data(covid_dataset, KEY_COUNTIRES, TIME_SERIES_DAYS, PREDICT_DAYS)

# Train Dataset of all countries
X_train_all, Y_train_all = covid.train_data(covid_dataset, COUNTRIES, TIME_SERIES_DAYS, PREDICT_DAYS)

X_test, Y_test = covid.train_data(covid_dataset, ["United Kingdom", "Iraq", "Canada", ], TIME_SERIES_DAYS, PREDICT_DAYS)

X_train_key, Y_train_key = X_train_key / K, Y_train_key / K
X_train_all, Y_train_all = X_train_all / K, Y_train_all / K
X_test, Y_test = X_test / K, Y_test / K
# Models (Key and All)
covid_model_key = covid.build_lstm_concatenate(X_train_key.shape, Y_train_key.shape)
covid_model_all = covid.build_lstm_concatenate(X_train_all.shape, Y_train_all.shape)

best_checkpoint_key = ModelCheckpoint(generate_checkpoint_name(covid.covid_dataset.dates[-1], "key"), save_best_only=True)
best_checkpoint_all = ModelCheckpoint(generate_checkpoint_name(covid.covid_dataset.dates[-1], "all"), save_best_only=True)

print("\n================")
print("Training model on key countries dataset")
print("================\n")

# Key training
history_key = covid_model_key.fit(
    X_train_key, Y_train_key,
    validation_data=[X_test, Y_test],
    batch_size=64,
    verbose=2,
    epochs=70
)

print("\n================")
print("Training model on all countries dataset")
print("================\n")

# All training
history_all = covid_model_all.fit(
    X_train_all, Y_train_all,
    validation_data=[X_test, Y_test],
    batch_size=512,
    verbose=2,
    epochs=70,
)

show_train_plot(history=history_key.history)
show_train_plot(history=history_all.history)

covid_model_key.save(generate_model_name(dates[-1], "key"))
covid_model_all.save(generate_model_name(dates[-1], "all"))

