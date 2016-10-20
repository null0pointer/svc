from sklearn import svm
import csv_parser

print('Parsing training data...')
training_features, training_targets = csv_parser.parse_training_csv('numerai_training_data.csv')

# Train the model
print('Training model...')
classifier = svm.SVC(probability=True, verbose=True)
classifier.fit(training_features, training_targets)

print('Parsing test data...')
prediction_ids, prediction_features = csv_parser.parse_prediction_csv('numerai_tournament_data.csv')
        
# Predict the outcomes
print('Running predictions...')
predictions = classifier.predict_proba(prediction_features)

# Set up the output array
prediction_output = [['t_id', 'probability']]

# Populate the output array
for i in range(len(prediction_ids)):
    tid = prediction_ids[i]
    prediction = predictions[i][1]
    prediction_output.append([tid, prediction])
    
print('Writing predictions to csv...')
csv_parser.write_lines_to_csv('predictions.csv', prediction_output)