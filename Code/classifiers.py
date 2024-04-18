import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models, callbacks, metrics, losses

import utils


def tensorflow_fcnn(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                    y_val: pd.DataFrame) -> np.ndarray:
    """
    Trains a fcnn using tensoflow and validate its performances. \n
    Edit the network architecture directly here.
    :return: The predicted most probable class for each X_val sample
    """
    # Define your neural network architecture
    model = models.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')  # Adjust num_classes according to your problem
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Define callbacks (optional)
    early_stopping = callbacks.EarlyStopping(patience=3)

    # Train the model
    history = model.fit(X_train, y_train, epochs=300,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    # Evaluate the model on the validation set
    perf = model.evaluate(X_val, y_val, return_dict=True, verbose=2)
    print(perf)
    res: np.ndarray = model.predict(X_val, verbose=0)
    return res.argmax(axis=1)


def sklearn_rf(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_classifier.fit(X_train, y_train)
    y_val_pred = rf_classifier.predict(X_val)
    print(f'Accuracy : {np.mean(y_val == y_val_pred)}\n')
    feature_importance_sort = rf_classifier.feature_importances_.argsort()
    plt.barh(X_train.columns[feature_importance_sort][-20:], rf_classifier.feature_importances_[feature_importance_sort][-20:])

    return y_val_pred
