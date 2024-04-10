import pandas as pd

dataPath = '../Classification Music/'
firstFeatures = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"]
firstClasses = ["pop", "disco", "metal", "classical"]
filenames = [f'{dataPath}GenreClassData_{i}s.txt' for i in [5, 10, 30]]


def load_data():
    df5s = pd.read_csv(filenames[0], sep='\t')
    df10s = pd.read_csv(filenames[1], sep='\t')
    df30s = pd.read_csv(filenames[2], sep='\t')
    return df5s, df10s, df30s


def task1_df(df: pd.DataFrame):
    return df[firstFeatures].groupby('Genre').agg(list).loc[firstClasses]


