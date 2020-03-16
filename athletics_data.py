import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def make_url(page_nr):
    return "https://www.worldathletics.org/records/toplists/sprints/100-metres/outdoor/men/senior/2019?regionType=world" \
           "&timing=electronic&windReading=regular&page=%s&bestResultsOnly=false" % page_nr


def trim_rows(rows):
    return [[el.strip() for el in row] for row in rows]


def strip_data(page_limit):
    all_rows = []
    for i in range(1, page_limit + 1):
        url = make_url(i)
        html_data = requests.get(url)
        html_bs = BeautifulSoup(html_data.text)
        table_data = [[cell.text for cell in row("td")] for row in html_bs("tr")]
        trimmed_rows = trim_rows(table_data[1:])
        all_rows += trimmed_rows
    return all_rows


def save_athletics_data():
    data = strip_data(255)
    df = pd.DataFrame.from_records(data)
    df.columns = ['rank', 'mark', 'wind', 'competitor', 'dob', 'nat', 'pos', 'empty_col', 'venue', 'date', 'results_score']
    df.drop('empty_col', axis=1)
    df.to_csv("100m_men_2019.csv", header=True, index=False, sep=",")


def plot_score_hist(data, col_name):
    from sklearn.neighbors import KernelDensity

    non_nan = data[data[col_name].notnull()][col_name].to_numpy().reshape(-1, 1)
    minimum = np.min(non_nan)
    maximum = np.max(non_nan)
    ran = maximum - minimum
    xs = np.linspace((minimum - 0.05 * ran), (maximum + 0.05 * ran), 300)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(non_nan)
    log_dens = kde.score_samples(xs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(xs, np.exp(log_dens))


speed_df = pd.read_csv('/home/data/PycharmProjects/ideas/normal_skew/100m_men_2019.csv', sep=',')
speed_std = speed_df[['mark', 'competitor']].groupby('competitor').agg([np.mean, lambda x: np.std(x, ddof=0)])
speed_std.columns = ['mean', 'std']


plot_score_hist(speed_std[speed_std['mean'] > 0.0], 'mean')
plt.show()

