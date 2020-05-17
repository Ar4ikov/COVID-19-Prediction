# | Created by Ar4ikov
# | Время: 16.05.2020 - 13:53

from coronavirus_dataset.generate_dataset import CovidDataset
from matplotlib import pyplot as plt
from matplotlib import image as plt_img
import matplotlib as mpl
import numpy as np
from PIL import Image


def temperature_map(data, days):
    to_pixel = lambda x, max_: [255 * (x / max(1, max_)),
                                255 * (x / max(1, max_)),
                                255 * (x / max(1, max_))]

    imgplot = []

    for i in range(len(days)):
        width = []
        for country in data.values():
            max_ = max(country["daily"]["confirmed"])
            width.append(to_pixel(country["daily"]["confirmed"][i], max_))

        imgplot.append(width)

    img = Image.fromarray(np.array(imgplot, dtype="uint8"))
    img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

    img.save("normal.png")

    plot_img = plt_img.imread("normal.png")
    plt.imshow(plot_img[:, :, 0], cmap="RdPu_r")

    max_value = max([max(x["daily"]["confirmed"]) for x in data.values()])
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels([x for x in range(0, max_value + 1, 6000)])
    cbar.ax.tick_params(labelsize=18)

    plt.gcf().set_size_inches(7, 10)

    plt.show()


def show_statistic_plot(country):
    statistic = dataset[country]

    def start_from(arr):
        for idx, i in enumerate(arr):
            if i > 40:
                return idx

    total_cases = statistic["all_time"]["confirmed"]
    start_index = start_from(total_cases)

    total_cases = total_cases[start_index:]
    new_cases = statistic["daily"]["confirmed"][start_index:]

    fig, (new, total) = plt.subplots(2, 1)
    fig.set_size_inches(12, 16)

    new.bar(["/".join(x.split("/")[:2]) for x in dates[start_index:]], new_cases,
            lw=1, color="#30c9e6", label="Новые случаи заражения")
    total.bar(["/".join(x.split("/")[:2]) for x in dates[start_index:]], total_cases,
              lw=1, color="#1aa5b8", label="Новые случаи заражения")

    new.set_title(f"Динамика COVID-19 в {country}", fontsize=20)
    total.set_title(f"Суммарное количество случаев COVID-19 в {country}", fontsize=20)

    new.set(xlabel="Дни", ylabel="# Случаев")
    total.set(xlabel="Дни", ylabel="# Случаев")

    new.tick_params(axis="x", rotation=90)
    total.tick_params(axis="x", rotation=90)

    mpl.rc('xtick', labelsize=8)
    mpl.rc('ytick', labelsize=16)

    plt.show()


covid_dataset = CovidDataset()
covid_dataset.generate_dataset()
dataset = covid_dataset.dataset
dates = covid_dataset.dates

STATISTIC_COUNTRIES = ["Russia", "US", "Canada", "United Kingdom", "China", "Korea, South", "Japan", "Greece", "Israel",
                       "Brazil", "Spain", "Italy", "Germany", "France", "Sweden", "Peru", "Ukraine", "Belarus"]

temperature_map(dataset, dates)

for country in STATISTIC_COUNTRIES:
    show_statistic_plot(country)
