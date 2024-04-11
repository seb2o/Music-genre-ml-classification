import pandas as pd
from matplotlib import pyplot as plt

dataPath = '../Classification Music/'
givenFeatures = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
pickedFeatures = ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo", "Genre"]
givenClasses = ["pop", "disco", "metal", "classical"]
filenames = [f'{dataPath}GenreClassData_{i}s.txt' for i in [5, 10, 30]]
columns_to_drop = ['File']


def load_data(filename: str, dropType=True) -> pd.DataFrame:
    """
    Fetch dataset from their .txt file into a nice dataframe.
    Normally should not be used outside this file. Prefer :func:`utils.task4_df() <utils.task4_df>`
    :param filename: The .txt file used
    :param dropType: if True, drop the "Type" column of the dataset
    :return: dataframe of one of the dataset
    """
    if dropType:
        columns_to_drop.append('Type')
    return pd.read_csv(filename, sep='\t').drop(columns=columns_to_drop)


def task1_df(dropType=False) -> pd.DataFrame:
    """
    Option to drop type column to ease EDA. \n
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    :return: df with the above features, from the 30s dataset
    """
    insideGivenFeatures = givenFeatures.copy()
    if not dropType:
        insideGivenFeatures += ['Type']
    return load_data(filenames[2], dropType=dropType)[insideGivenFeatures]


def task2_df() -> pd.DataFrame:
    """
    Features :\n
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"] \n
    Classes :\n
    ["pop", "disco", "metal", "classical"]
    :return: df with the above features, only for the 4 above classes, from the 30s dataset
    """
    return task1_df().groupby('Genre').agg(list).loc[givenClasses]


def task3_df(dropType=False) -> pd.DataFrame:
    """
    Option to drop "Type" column to ease EDA.\n
    Features :\n
    ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo", "Genre"]
    :return: df from the 30s dataset, with only the 4 handpicked features above
    """
    insidePickedFeatures = pickedFeatures.copy()
    if not dropType:
        insidePickedFeatures += ['Type']
    return load_data(filenames[3], dropType=dropType)[insidePickedFeatures]


def task4_df(dropType=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the whole dataset into 3 separate data frames
    :param dropType: if True, the "Type" column is dropped
    :return: a 3-uple with, in this order, df5s, df10s, df30s.
    """
    df5s = load_data(filenames[0], dropType=dropType)
    df10s = load_data(filenames[1], dropType=dropType)
    df30s = load_data(filenames[2], dropType=dropType)
    return df5s, df10s, df30s


def plot_feature_by_genre(feature_to_compare, use_df) -> None:
    """
    Group the df rows by Genre and keeps only values of specified column. \n
    Plot values for each genre in a separate plot with shared x axis for easier comparison
    :param feature_to_compare: feature that will be plotted
    :param use_df: either df5, df10 of df30
    """
    grouped = use_df.groupby('Genre')[feature_to_compare]
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(15, 10))
    ax_index = 0
    for name, values in grouped:
        axes[ax_index // 4][ax_index % 4].hist(values, density=True)
        axes[ax_index // 4][ax_index % 4].set_title(name)
        ax_index += 1


def plot_corr(df, topRotation=90, figNumber=None) -> None:
    """
    Plot correlation heatmap of given dataframe
    :param df: the data frame we want to plot
    :param topRotation: the rotation of the top legend
    :param figNumber: used to allow creating a figure before calling this function. Useful for enforcing large figures.
    """
    plt.matshow(df.corr(), fignum=figNumber)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=topRotation)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
