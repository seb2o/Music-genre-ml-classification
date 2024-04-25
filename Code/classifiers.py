import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

import utils


def tensorflow_fcnn(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                    y_val: pd.DataFrame, verbose=True) -> np.ndarray:
    """
    Trains a fcnn using tensoflow and validate its performances. \n
    Edit the network architecture directly here.
    :return: The predicted most probable class for each X_val sample
    """

    if not verbose:
        tf.get_logger().setLevel('ERROR')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', ),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs=10,
              validation_data=(X_val, y_val),
              verbose=2 if verbose else 0
              )

    res: np.ndarray = model.predict(X_val, verbose=0)
    return res.argmax(axis=1)


def sklearn_rf(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_classifier.fit(X_train, y_train)
    y_val_pred = rf_classifier.predict(X_val)
    print(f'Accuracy : {np.mean(y_val == y_val_pred)}\n')
    feature_importance_sort = rf_classifier.feature_importances_.argsort()
    plt.barh(X_train.columns[feature_importance_sort][-20:],
             rf_classifier.feature_importances_[feature_importance_sort][-20:])

    return y_val_pred


def build_one_lstm(x_train, y_train, x_test, y_test) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=15,
        validation_data=(x_test, y_test),
        verbose=0
    )
    return model


def build_ensemble_lstm(df, nmodels=5):
    """
    Trains lstm models on the df and return them as a list
    :param df: the dataset df, which *will* be pre processed
    :param nmodels: the number of models that will be trained
    :return: a list of the models
    """
    models = []
    for step in range(nmodels):
        print(f"Training model {step + 1}")
        x_train, y_train, x_test, y_test = utils.preproccess_for_lstm(df)
        curr_model = build_one_lstm(x_train, y_train, x_test, y_test)
        models.append(curr_model)
    return models


def predict_ensemble_lstm(x_test, models):
    """
    For each model on the list, make predictions on the sequence. Then the most voted label for each sequence
    is stored in the "combined" column
    :param x_test:
    :param models:
    :return:
    """
    tf.get_logger().setLevel('ERROR')
    all_predictions = pd.DataFrame(
        [
            [
                np.argmax(track_pred_probas[-1])
                for track_pred_probas in model.predict(x_test, verbose=0)
            ]
            for model in models
        ]
    ).transpose()
    all_predictions["combined"] = pd.DataFrame(
        all_predictions
        .apply(lambda row: row.mode()[0], axis=1))

    return all_predictions
