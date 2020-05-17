# | Created by Ar4ikov
# | Время: 13.05.2020 - 23:29

from os import path
from statistics import stdev
import pandas as pd
import numpy as np
from json import dumps, loads
from sys import stdout


class CovidDataset:
    def __init__(self):
        """Получение статистики из репозитория CSSE of John's Hopkins University"""

        self.confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
        self.deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        self.recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

        """Статистика по США"""
        self.latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/05-03-2020.csv')
        self.us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/05-03-2020.csv')

        cols = self.confirmed_df.keys()

        """Преобразование файлов в нужный вид"""
        self.confirmed = self.confirmed_df.loc[:, cols[4]:cols[-1]].fillna(0)
        self.deaths = self.deaths_df.loc[:, cols[4]:cols[-1]].fillna(0)
        self.recoveries = self.recoveries_df.loc[:, cols[4]:cols[-1]].fillna(0)

        # Даты
        self.dates = self.confirmed.keys()[:-2]

        # Страны
        self.countries = sorted(list({x for x in self.confirmed_df["Country/Region"]}))
        self.KEY_COUNTRIES = ["China", "Korea, South", "Japan", "Greece", "Italy", "Spain", "Czechia", "Israel", "France", "Germany"]

        self.dataset = {}
        self.filename = f"covid-19-{self.dates[-1].replace('/', '-')}.json"

    @staticmethod
    def daily_increase(data):
        """Дневной прирост"""
        d = []
        for i in range(len(data)):
            if i == 0:
                d.append(data[0])
            else:
                d.append(data[i] - data[i - 1])
        return d

    @staticmethod
    def mortality_and_recovery_rate(confirmed_, recoveries_, deaths_):
        """Летальность и коэффициент выздоравлений"""
        mortality = []
        recovery = []

        for i in range(len(confirmed_)):
            delimeter = confirmed_[i]

            if delimeter == 0:
                mortality.append(0), recovery.append(0)
                continue

            mortality.append(deaths_[i] / confirmed_[i])
            recovery.append(recoveries_[i] / confirmed_[i])

        return mortality, recovery

    @staticmethod
    def stdev_daily(confirmed_, recoveries_, deaths_):
        """Среднеквадратичное отклонение случаев заражения, выздоравления и смертей"""
        stdev_confirmed = []
        stdev_recoveries = []
        stdev_deaths = []

        for i in range(len(confirmed_)):
            if i < 2:
                stdev_confirmed.append(0)
                stdev_recoveries.append(0)
                stdev_deaths.append(0)
                continue

            stdev_confirmed.append(stdev(confirmed_[:i]))
            stdev_recoveries.append(stdev(recoveries_[:i]))
            stdev_deaths.append(stdev(deaths_[:i]))

        return stdev_confirmed, stdev_recoveries, stdev_deaths

    def generate_dataset(self) -> dict:
        if path.isfile(self.filename):
            with open(self.filename, "r") as file_:
                self.dataset = loads(file_.read())
                return self.dataset

        for i in self.dates:
            stdout.write(f"{i}  \r")

            for country in self.countries:
                if country not in self.dataset:
                    self.dataset[country] = {
                        "all_time": {
                            "confirmed": [],
                            "recoveries": [],
                            "deaths": []
                        },
                        "daily": {
                            "confirmed": [],
                            "recoveries": [],
                            "deaths": []
                        },
                        "statistic": {
                            "recovery_rate": [],
                            "mortality_rate": [],
                            "stdev": {
                                "confirmed": [],
                                "recoveries": [],
                                "deaths": []
                            }
                        }
                    }

                self.dataset[country]["all_time"]["confirmed"].append(
                    self.confirmed_df[self.confirmed_df["Country/Region"] == country][i].sum())
                self.dataset[country]["all_time"]["recoveries"].append(
                    self.recoveries_df[self.recoveries_df["Country/Region"] == country][i].sum())
                self.dataset[country]["all_time"]["deaths"].append(self.deaths_df[self.deaths_df["Country/Region"] == country][i].sum())

        for country in self.countries:
            self.dataset[country]["daily"]["confirmed"] = self.daily_increase(self.dataset[country]["all_time"]["confirmed"])
            self.dataset[country]["daily"]["recoveries"] = self.daily_increase(self.dataset[country]["all_time"]["recoveries"])
            self.dataset[country]["daily"]["deaths"] = self.daily_increase(self.dataset[country]["all_time"]["deaths"])

            # Дневные случаи
            per_cases = self.dataset[country]["daily"]["confirmed"]
            per_recoveries = self.dataset[country]["daily"]["recoveries"]
            per_deaths = self.dataset[country]["daily"]["deaths"]

            # Расчет дополнительной статистики
            per_mor, per_rec = self.mortality_and_recovery_rate(per_cases, per_recoveries, per_deaths)
            stdev_cases, stdev_rec, stdev_death = self.stdev_daily(per_cases, per_recoveries, per_deaths)

            self.dataset[country]["statistic"]["mortality_rate"] = per_mor
            self.dataset[country]["statistic"]["recovery_rate"] = per_rec

            self.dataset[country]["statistic"]["stdev"]["confirmed"] = stdev_cases
            self.dataset[country]["statistic"]["stdev"]["recoveries"] = stdev_rec
            self.dataset[country]["statistic"]["stdev"]["deaths"] = stdev_death

        with open(self.filename, "w") as file_:
            file_.write(dumps(self.dataset, default=lambda x: int(x)))

        return self.dataset

    def days_since_1_22(self):
        return np.array([i for i in range(len(self.dates))]).reshape(-1, 1)
