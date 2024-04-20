import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sn

dataPath = '../Classification Music/'
metaFeatures = ["Type", "GenreID"]
EDAfeatures = ['Genre']
task1_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
task3_features = ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo"]
rfPickedFeatures = ["rmse_mean", "rmse_var", "spectral_contrast_var", "mfcc_4_mean"]
mrmr_features = [
    'mfcc_5_std',
    'mfcc_4_mean',
    'mfcc_1_mean',
    'spectral_bandwidth_mean',
    'spectral_contrast_var',
    'rmse_var',
    'spectral_rolloff_mean',
    'spectral_centroid_var',
    'mfcc_6_mean',
    'spectral_centroid_mean',
    'mfcc_7_std',
    'mfcc_8_mean',
    'spectral_rolloff_var',
    'rmse_mean',
    'mfcc_2_mean',
    'mfcc_4_std',
    'chroma_stft_2_mean',
    'mfcc_6_std',
    'mfcc_9_mean',
    'chroma_stft_5_mean',
    'chroma_stft_7_mean',
    'tempo',
    'spectral_flatness_var',
    'chroma_stft_9_mean',
    'mfcc_3_std',
    'zero_cross_rate_mean',
    'mfcc_12_mean',
    'spectral_contrast_mean',
    'zero_cross_rate_std',
    'spectral_bandwidth_var',
    'chroma_stft_4_mean',
    'mfcc_7_mean',
    'chroma_stft_12_mean',
    'mfcc_3_mean',
    'spectral_flatness_mean',
    'mfcc_8_std',
    'chroma_stft_10_mean',
    'mfcc_10_mean',
    'mfcc_10_std']
task2_classes = ["pop", "disco", "metal", "classical"]
filenames = [f'{dataPath}GenreClassData_{i}s.txt' for i in [5, 10, 30]]
columns_to_drop = ['File']
genreNames = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock",
              "hihop", "country", "jazz"]


def load_data(filename: str) -> pd.DataFrame:
    """
    Fetch dataset from their .txt file into a nice dataframe.
    Normally should not be used outside this file. Prefer :func:`utils.task4_df() <utils.task4_df>`
    :param filename: The .txt file used
    :return: dataframe of one of the dataset
    """
    return pd.read_csv(filename, sep='\t').drop(columns=columns_to_drop).rename(columns={"Track ID": "TrackID"})


def task1_df(isEDA=False) -> pd.DataFrame:
    """
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    :param isEDA: if True, remove non meaningful features to ease EDA
    :return: df with the above features, from the 30s dataset
    """
    additionalFeatures = EDAfeatures if isEDA else metaFeatures
    return load_data(filenames[2])[task1_features + additionalFeatures]


def task2_df() -> pd.DataFrame:
    """
    Features :\n
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"] \n
    Classes :\n
    ["pop", "disco", "metal", "classical"]\n
    The returned df has a row for each genre, and a column for each feature.
    A cell contains a list of its feature's value for all samples of the dataset of genre corresponding
    to the row.

    :return: df with the above features, only for the 4 above classes, from the 30s dataset.
    """
    return task1_df(isEDA=True).groupby('Genre').agg(list).loc[task2_classes]


def task3_df(isEDA=False, pickedFeatures=None) -> pd.DataFrame:
    """
    Option to drop "Type" column to ease EDA.\n
    Features :\n
    ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo", "Genre"]
    :param isEDA: if True, remove non meaningful features to ease EDA
    :return: df from the 30s dataset, with only the 4 handpicked features above
    """
    if pickedFeatures is None:
        pickedFeatures = task3_features

    additionalFeatures = EDAfeatures if isEDA else metaFeatures
    return load_data(filenames[2])[pickedFeatures + additionalFeatures]


def task4_df(isEDA=False) -> tuple[DataFrame, ...]:
    """
    Loads the whole dataset into 3 separate data frames. Drops by default 'Genre' and 'Track ID' columns.
    :param isEDA: if set to True, remove the 'Type' column.
    :return: a 3-uple with, in this order, df5s, df10s, df30s.
    """
    featureToDrop = ['Genre']
    if isEDA:
        featureToDrop = ['Type']
    return tuple(load_data(filename).drop(columns=featureToDrop) for filename in filenames)


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


def plot_corr(df, annot=True, figsize=None, abs=True, keep_diag=False) -> None:
    """
    Plot correlation heatmap of given dataframe. \n
    :param keep_diag: if False, remove the diagonal of 1 in the correlation plot
    :param abs: if True, plot the absolute values of the correlations
    :param figsize: if specified, create a figure of this size
    :param annot: if True, add correlation value on each cell
    :param df: the data frame we want to plot
    """
    if figsize is not None:
        plt.figure(figsize=figsize)

    corr = df.corr().abs() if abs else df.corr()


    mask = np.triu(np.ones_like(corr))

    if keep_diag:
        np.fill_diagonal(mask, False)

    sn.set(font_scale=1.4)
    sn.heatmap(
        corr,
        annot=annot,
        annot_kws={"size": 16},
        linewidths=.5,
        mask=mask,
    )


def train_val_split(df: DataFrame, keep_trackID=False) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits an input df into features/target dataframes, for train and test. \n
    Normalize the features according to the distribution of the training set, and shuffle the dataset
    :param keep_trackID: if False, drop the trackID column. Else keep it in y_test
    as an additional column
    :param df: should have a column "Type" and a column "GenreID" (target variable)
    :return: X_train, y_train, X_test, y_test
    """
    hasTrackID = 'TrackID' in df.columns
    features_to_drop = ['GenreID', 'TrackID'] if hasTrackID else 'GenreID'
    y_test_features = ['GenreID', 'TrackID'] if hasTrackID and keep_trackID else 'GenreID'

    df = df.sample(frac=1)
    isTrain = df['Type'] == 'Train'

    dfTrain = df[isTrain].drop(columns='Type')
    dfTest = df[~isTrain].drop(columns='Type')


    X_train = dfTrain.drop(columns=features_to_drop, errors='ignore')
    y_train = dfTrain['GenreID']

    X_test = dfTest.drop(columns=features_to_drop, errors='ignore')
    y_test = dfTest[y_test_features]


    X_train_scaled = (X_train - X_train.mean()) / X_train.std()
    X_test_scaled = (X_test - X_train.mean()) / X_train.std()

    return X_train_scaled, y_train, X_test_scaled, y_test


def preproccess_for_lstm(df):
    """
    reshape, shuffles and normalize dataframe into a list of sequence of sample ( each sequence is a track)
    :param df: preferably df5s
    :return: x_train, y_train, x_test, y_test
    """

    isTrain = df['Type'] == 'Train'
    dfTrain = df[isTrain].drop(columns='Type')
    dfTest = df[~isTrain].drop(columns='Type')

    # columns_to_scale = dfTrain.columns.difference(['GenreID', 'TrackID'])
    columns_to_scale = mrmr_features
    dfTrain_scaled = (dfTrain[columns_to_scale] - dfTrain[columns_to_scale].mean()) / dfTrain[columns_to_scale].std()
    dfTest_scaled = (dfTest[columns_to_scale] - dfTrain[columns_to_scale].mean()) / dfTrain[columns_to_scale].std()

    dfTrain_scaled["GenreID"] = dfTrain["GenreID"]
    dfTrain_scaled["TrackID"] = dfTrain["TrackID"]
    dfTest_scaled["GenreID"] = dfTest["GenreID"]
    dfTest_scaled["TrackID"] = dfTest["TrackID"]

    train_by_track = dfTrain_scaled \
        .groupby('TrackID') \
        .apply(lambda group_df:
               (group_df.iloc[:, :-1].values, group_df.iloc[:, -1].values),
               include_groups=False) \
        .sample(frac=1)
    x_train = np.stack(train_by_track.apply(lambda x: x[0]))
    y_train_redundant = np.stack(train_by_track.apply(lambda x: x[1]))
    y_train = train_by_track.apply(lambda x: x[1][0])

    test_by_track = dfTest_scaled \
        .groupby('TrackID') \
        .apply(lambda group_df:
               (group_df.iloc[:, :-1].values, group_df.iloc[:, -1].values),
               include_groups=False) \
        .sample(frac=1)
    x_test = np.stack(test_by_track.apply(lambda x: x[0]))
    y_test_redundant = np.stack(test_by_track.apply(lambda x: x[1]))
    y_test = test_by_track.apply(lambda x: x[1][0])

    return x_train, y_train_redundant, x_test, y_test_redundant
