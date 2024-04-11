import pandas as pd
from matplotlib import pyplot as plt

dataPath = '../Classification Music/'
givenFeatures = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
pickedFeatures = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
givenClasses = ["pop", "disco", "metal", "classical"]
filenames = [f'{dataPath}GenreClassData_{i}s.txt' for i in [5, 10, 30]]


def load_data(filename, dropType=True):
    columns_to_drop = ['File']
    if dropType:
        columns_to_drop.append('Type')
    return pd.read_csv(filename, sep='\t').drop(columns=columns_to_drop)


def task1_df():
    """
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    :return: pd.Dataframe df with the above features, from the 30s dataset
    """
    return load_data(filenames[2])[givenFeatures]


def task2_df():
    """
    Features :["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    Classes : ["pop", "disco", "metal", "classical"]
    :return: pd.Dataframe df with the above features, only for the 4 above classes, from the 30s dataset
    """
    return task1_df().groupby('Genre').agg(list).loc[givenClasses]


def task3_df():
    """
    Features :["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    :return: df from the 30s dataset, with only the 4 handpicked features above
    """
    return load_data(filenames[3])[pickedFeatures]


def task4_df():
    df5s = load_data(filenames[0])
    df10s = load_data(filenames[1])
    df30s = load_data(filenames[2])
    return df5s, df10s, df30s


def plot_feature_by_genre(feature_to_compare, use_df):
    grouped = use_df.groupby('Genre')[feature_to_compare]
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(15, 10))
    ax_index = 0
    for name, values in grouped:
        axes[ax_index // 4][ax_index % 4].hist(values, density=True)
        axes[ax_index // 4][ax_index % 4].set_title(name)
        ax_index += 1


def plot_corr(df, topRotation=90, figNumber=None):
    plt.matshow(df.corr(), fignum=figNumber)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=topRotation)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)