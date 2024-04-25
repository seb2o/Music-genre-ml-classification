from Code import utils, classifiers, evaluations

filename = '../Classification Music/GenreClassData_5s.txt'

df = utils.load_data(filename).drop(columns='Genre')

# Train the models on the 5s dataset
models = classifiers.build_ensemble_lstm(df, nmodels=15)

test_dataset_file = '../Classification Music/GenreClassData_5s.txt'
df = utils.load_data(test_dataset_file).drop(columns='Genre')
_, _, x_test, _ = utils.preproccess_for_lstm(df, randomize_test=False)

ensemble_prediction = classifiers.predict_ensemble_lstm(x_test, models)
prediction = ensemble_prediction.combined.values
# The printed prediction is in the same order as the input dataframe
print(prediction)
