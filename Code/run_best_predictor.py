from Code import utils, classifiers, evaluations

# Replace filename by path of the test data
filename = '../Classification Music/GenreClassData_5s.txt'

# Here the columns 'File' is removed in load_data()
df = utils.load_data(filename).drop(columns='Genre')

# Here columns TrackID and Type are assumed. If 100% of the dataframe is Test, comment the code for _train
# in preproccess_for_lstm()
_, _, x_test, y_test = utils.preproccess_for_lstm(df, randomize_test=False)
models = classifiers.build_ensemble_lstm(df, nmodels=15)
ensemble_prediction = classifiers.predict_ensemble_lstm(x_test, models)
prediction = ensemble_prediction.combined.values
# The printed prediction is in the same order as the input dataframe
print(prediction)
# print(evaluations.evaluate_ensemble_lstm(ensemble_prediction, y_test))
