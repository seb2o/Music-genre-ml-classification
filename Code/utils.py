import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

dataPath = '../Classification Music/'
metaFeatures = ["Type", "GenreID"]
EDAfeatures = ['Genre']
givenFeatures = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
pickedFeatures = ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo"]
rfPickedFeatures = ["rmse_mean", "rmse_var", "spectral_contrast_var", "mfcc_4_mean"]
givenClasses = ["pop", "disco", "metal", "classical"]
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
    return pd.read_csv(filename, sep='\t').drop(columns=columns_to_drop)


def task1_df(isEDA=False) -> pd.DataFrame:
    """
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
    :param isEDA: if True, remove non meaningful features to ease EDA
    :return: df with the above features, from the 30s dataset
    """
    additionalFeatures = EDAfeatures if isEDA else metaFeatures
    return load_data(filenames[2])[givenFeatures + additionalFeatures]


def task2_df() -> pd.DataFrame:
    """
    Features :\n
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"] \n
    Classes :\n
    ["pop", "disco", "metal", "classical"]
    :return: df with the above features, only for the 4 above classes, from the 30s dataset
    """
    return task1_df(isEDA=True).groupby('Genre').agg(list).loc[givenClasses]


def task3_df(isEDA=False) -> pd.DataFrame:
    """
    Option to drop "Type" column to ease EDA.\n
    Features :\n
    ["zero_cross_rate_mean", "spectral_rolloff_mean", "mfcc_1_mean", "tempo", "Genre"]
    :param isEDA: if True, remove non meaningful features to ease EDA
    :return: df from the 30s dataset, with only the 4 handpicked features above
    """
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


def plot_corr(df, topRotation=90, figNumber=None) -> None:
    """
    Plot correlation heatmap of given dataframe
    :param df: the data frame we want to plot
    :param topRotation: the rotation of the top legend
    :param figNumber: used to allow creating a figure before calling this function. Useful for enforcing large figures.
    """
    plt.matshow(df.corr(), fignum=figNumber)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=topRotation)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)


def train_val_split(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits an input df into features/target dataframes, for train and test. \n
    Normalize the features according to the distribution of the training set, and shuffle the dataset \n
    The returned y_test has a column GenreID and a column TrackID. use y_test.GenreID to get target values.
    :param df: should have a column "Type" and a column "GenreID" (target variable)
    :return: X_train, y_train, X_test, y_test
    """
    df = df.sample(frac=1)

    isTrain = df['Type'] == 'Train'
    dfTrain = df[isTrain].drop(columns='Type')
    dfTest = df[~isTrain].drop(columns='Type')

    X_train = dfTrain.drop(columns=['GenreID', 'Track ID'])
    y_train = dfTrain['GenreID']

    X_test = dfTest.drop(columns=['GenreID', 'Track ID'])
    y_test = dfTest[['GenreID', 'Track ID']]
    y_test = y_test.rename(columns={'Track ID': 'TrackID'})

    X_train_scaled = (X_train - X_train.mean()) / X_train.std()
    X_test_scaled = (X_test - X_train.mean()) / X_train.std()

    return X_train_scaled, y_train, X_test_scaled, y_test
