#!/usr/local/bin/python
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

covid_data = "./COVID-19/csse_covid_19_data/csse_covid_19_daily_reports"

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


def main(data_path):
    covid_labels = [
        CovidLabel('Australia', 2.54e7, 'Australia'),
        CovidLabel('China', 1.44e9, ['China', 'Mainland China']),
        CovidLabel('France', 6.52e7, 'France'),
        CovidLabel('Germany', 8.37e7, 'Germany'),
        CovidLabel('Italy', 6.05e7, 'Italy'),
        CovidLabel('Spain', 5.68e7, 'Spain'),
        CovidLabel('South Korea', 5.13e7, ['South Korea', 'Korea, South']),
        CovidLabel('USA', 3.31e8, 'US'),

        # CovidLabel('Singapore', 5.84e6, 'Singapore'),
        # CovidLabel('San Marino', 3.39e4, 'San Marino'),
        # CovidLabel('Andorra', 7.72e4, 'Andorra'),
        # CovidLabel('AUS: NSW', 'Australia', 'New South Wales'),
        # CovidLabel('AUS: QLD', 'Australia', 'Queensland'),
        # CovidLabel('AUS: Vic', 'Australia', 'Victoria'),
        # CovidLabel('USA: NY', 'US', 'New York')
    ]

    results = pd.DataFrame(columns=result_columns)
    results = results.astype(result_column_types)
    results = results.set_index(result_index_columns)

    for entry in os.scandir(data_path):
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

    fig, ax = plt.subplots()

    for covid_label in [_ for _ in covid_labels if _.get_label() in results.index]:
        df = results.loc[covid_label.get_label()].copy(deep=False)

        scale_factor = 1e3 / covid_label.population

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

        # df['NEW_DEATHS'] = df['DEATHS'].diff()
        # df['NEW_DEATHS_ROLLING'] = df['NEW_DEATHS'].rolling(min_periods=1, window=7).sum()
        # df = df.fillna(0)
        #
        # df = df[df['DEATHS'] > 100]
        # df = df[df['NEW_DEATHS_ROLLING'] > 0]
        #
        # df['DEATHS'] = df['DEATHS'] * scale_factor
        # df['NEW_DEATHS_ROLLING'] = df['NEW_DEATHS_ROLLING'] * scale_factor
        #
        # x_axis = 'DEATHS'
        # y_axis = 'NEW_DEATHS_ROLLING'

        if len(df.index) > 0:
            ax = df.plot(ax=ax, kind='line', x=x_axis, y=y_axis, label=covid_label.display_name, loglog=True)

    plt.legend(loc='best')
    plt.xlabel("Confirmed Cases (per 1k)")
    plt.ylabel("New Cases in Last 7 Days (per 1k)")
    # plt.xlabel("Deaths (per 1k)")
    # plt.ylabel("New Deaths in Last 7 Days (per 1k)")
    plt.gcf().set_size_inches(2 * plt.gcf().get_size_inches())

    plt.savefig("current.png")
    plt.show()


class CovidLabel:
    display_name: str
    population: float
    first_level: list
    second_level: list

    def __init__(self, display_name, population, first_level, second_level=None):
        self.display_name = display_name
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
    main(covid_data)
