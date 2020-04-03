#!/usr/local/bin/python
import pandas as pd
import numpy as np
from datetime import timedelta
import os
from pathlib import Path
import matplotlib.pyplot as plt

covid_data_path = "./COVID-19/csse_covid_19_data/csse_covid_19_daily_reports"

covid_date_column = 'DATE'
covid_first_level_column = 'COUNTRY_REGION'
covid_second_level_column = 'PROVINCE_STATE'

covid_index_columns = [covid_first_level_column, covid_second_level_column]
covid_data_columns = ['CONFIRMED', 'DEATHS', 'RECOVERED']
covid_columns = [*covid_index_columns, *covid_data_columns]
covid_column_types = {
    **{column: np.unicode for column in covid_index_columns},
    **{column: np.uint64 for column in covid_data_columns}
}

result_index_columns = covid_index_columns
result_columns = [covid_date_column, *covid_columns]
result_column_types = {**{covid_date_column: np.datetime64}, **covid_column_types}


def main():
    cmap = plt.get_cmap("tab10")
    covid_labels = [
        # CovidLabel('Australia', cmap(0), 2.54e7, 'Australia'),
        # CovidLabel('China', cmap(1), 1.44e9, ['China', 'Mainland China']),
        # CovidLabel('France', cmap(2), 6.52e7, 'France'),
        # CovidLabel('Germany', cmap(3), 8.37e7, 'Germany'),
        # CovidLabel('Italy', cmap(4), 6.05e7, 'Italy'),
        # CovidLabel('Iran', cmap(5), 8.37e7, 'Iran'),
        # CovidLabel('Singapore', cmap(6), 5.84e6, 'Singapore'),
        # CovidLabel('Spain', cmap(7), 5.68e7, 'Spain'),
        # CovidLabel('South Korea', cmap(8), 5.13e7, ['South Korea', 'Korea, South']),
        # CovidLabel('USA', cmap(9), 3.31e8, 'US'),

        CovidLabel('Australia', cmap(0), 2.54e7, 'Australia'),
        CovidLabel('AUS: ACT', cmap(1), 4.26e5, 'Australia', 'Australian Capital Territory'),
        CovidLabel('AUS: NSW', cmap(2), 8.09e6, 'Australia', 'New South Wales'),
        CovidLabel('AUS: NT', cmap(3), 2.46e5, 'Australia', 'Northern Territory'),
        CovidLabel('AUS: SA', cmap(4), 1.75e6, 'Australia', 'South Australia'),
        CovidLabel('AUS: TAS', cmap(5), 5.34e6, 'Australia', 'Tasmania'),
        CovidLabel('AUS: QLD', cmap(6), 5.10e6, 'Australia', 'Queensland'),
        CovidLabel('AUS: VIC', cmap(7), 6.63e6, 'Australia', 'Victoria'),
        CovidLabel('AUS: WA', cmap(8), 2.62e6, 'Australia', 'Western Australia')
    ]

    results = prepare_results(covid_labels)

    start_date = pd.Timestamp(2020, 1, 25)
    stop_date = results[covid_date_column].max()
    current_date = start_date + timedelta(days=7)

    while current_date <= stop_date:
        create_plot(results, covid_labels, start_date, current_date)
        current_date += timedelta(days=1)


def create_plot(results, covid_labels, start_date, stop_date):
    fig, ax = plt.subplots()

    for covid_label in [_ for _ in covid_labels if _.get_label() in results.index]:
        scale_factor = 1e3 / covid_label.population

        df = results.loc[covid_label.get_label()].copy(deep=False)
        df = df[df['DATE'] >= start_date]
        df = df[df['DATE'] <= stop_date]

        df['NEW_CONFIRMED'] = df['CONFIRMED'].diff()
        df['NEW_CONFIRMED_ROLLING'] = df['NEW_CONFIRMED'].rolling(min_periods=1, window=7).sum()
        df = df.fillna(0)

        df = df[df['CONFIRMED'] > 100]
        df = df[df['NEW_CONFIRMED_ROLLING'] > 0]

        df['CONFIRMED'] = df['CONFIRMED'] * scale_factor
        df['NEW_CONFIRMED_ROLLING'] = df['NEW_CONFIRMED_ROLLING'] * scale_factor

        df = df[df['CONFIRMED'] > 1e-3]

        x_axis = 'CONFIRMED'
        y_axis = 'NEW_CONFIRMED_ROLLING'

        label = covid_label.display_name
        color = covid_label.color_name

        # if len(df.index) > 0:
        ax = df.plot(ax=ax, kind='line', x=x_axis, y=y_axis, color=color, label=label, loglog=True)

    timestamp = stop_date.strftime('%Y-%m-%d')

    plt.axis([1e-3, 5e0, 1e-4, 5e0])
    plt.legend(loc='upper left')
    plt.xlabel("Confirmed Cases (per 1k) - " + timestamp)
    plt.ylabel("New Cases in Last 7 Days (per 1k)")
    plt.gcf().set_size_inches(2 * plt.gcf().get_size_inches())

    plt.savefig("covid-19_confirmed-cases_" + timestamp + ".png")
    plt.close(fig)


def prepare_results(covid_labels):
    results = pd.DataFrame(columns=result_columns)
    results = results.astype(result_column_types)
    results = results.set_index(result_index_columns)

    for entry in os.scandir(covid_data_path):
        if not entry.is_file() or not entry.path.endswith('.csv'):
            continue

        file = entry.path
        date = pd.to_datetime(Path(file).stem)

        df = pd.read_csv(file)
        df = df.rename(columns=clean_column_names)
        df = df[covid_columns]
        df[covid_index_columns] = df[covid_index_columns].fillna('')
        df[covid_data_columns] = df[covid_data_columns].fillna(0)
        df = df.astype(covid_column_types)
        df = df.set_index(covid_index_columns)
        df = df.sort_index(level=df.index.names)

        for covid_label in covid_labels:
            index = df.index.get_level_values(covid_first_level_column) if covid_label.is_first_level() else df.index
            labels = [_ for _ in covid_label.get_lookup_labels() if _ in index]

            if labels:
                agg_data = [[date, *covid_label.get_label(), *df.loc[labels].sum(axis=0)]]
                agg_df = pd.DataFrame(data=agg_data, columns=result_columns)
                agg_df = agg_df.set_index(result_index_columns)

                results = results.append(agg_df)

    results = results.sort_values(by=covid_date_column)
    results = results.sort_index(level=results.index.names)

    return results


class CovidLabel:
    display_name: str
    color_name: str
    population: float
    first_level: list
    second_level: list

    def __init__(self, display_name, color_name, population, first_level, second_level=None):
        self.display_name = display_name
        self.color_name = color_name
        self.population = population

        if isinstance(first_level, str):
            self.first_level = [first_level]
        elif isinstance(first_level, list):
            self.first_level = first_level
        else:
            raise SystemExit('Error: unexpected first_level type: ', first_level)

        if second_level is None:
            self.second_level = []
        elif isinstance(second_level, str):
            self.second_level = [second_level]
        elif isinstance(second_level, list):
            self.second_level = second_level
        else:
            raise SystemExit('Error: unexpected second_level type: ', second_level)

    def is_first_level(self):
        return not self.second_level

    def get_label(self):
        if self.is_first_level():
            return self.first_level[0], ''
        else:
            return self.first_level[0], self.second_level[0]

    def get_lookup_labels(self):
        if self.is_first_level():
            return self.first_level
        else:
            return [(x, y) for x in self.first_level for y in self.second_level]


def clean_column_names(x):
    return x.translate({ord(y): '_' for y in '!@#$%^&*()[]{};:,./<>?\|`~-=_+'}).upper()


if __name__ == '__main__':
    main()
