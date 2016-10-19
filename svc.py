from sklearn import svm

# Set up arrays for features and targets
training_features = []
training_targets = []

# Open the training data CSV
f = open('numerai_training_data.csv', 'r')

# Split the CSV into lines
lines = f.read().split('\n')
f.close()

# Get the number of columns
columns = len(lines[0].split(','))

# Remove the title row from the lines
lines = lines[1:]

# Parse the lines into arrays and add to features and targets
print('Parsing...')
for line in lines:
    values = line.split(',')
    if len(values) == columns:
        target = float(values[-1])
        values = values[:-1]
        features = [float(x) for x in values]
        training_features.append(features)
        training_targets.append(target)

# Train the model
print('Training...')
classifier = svm.SVC(probability=True, verbose=True)
classifier.fit(training_features, training_targets)

# Set up arrays for predictions
prediction_ids = []
prediction_features = []

# Open the training data CSV
f = open('numerai_tournament_data.csv', 'r')

# Split the CSV into lines
lines = f.read().split('\n')
f.close()

# Get the number of columns
columns = len(lines[0].split(','))

# Remove the title row from the lines
lines = lines[1:]

# Parse the lines into arrays and add to features and ids
print('Parsing...')
for line in lines:
    values = line.split(',')
    if len(values) == columns:
        tid = str(values[0])
        values = values[1:]
        features = [float(x) for x in values]
        prediction_ids.append(tid)
        prediction_features.append(features)
        
# Predict the outcomes
predictions = classifier.predict_proba(prediction_features)

# Set up the output array
prediction_output = [['t_id', 'probability']]

# Populate the output array
for i in range(len(prediction_ids)):
    tid = prediction_ids[i]
    prediction = predictions[i][1]
    prediction_output.append([tid, prediction])
    
# Write to CSV
f = open('predictions.csv', 'w')
for prediction in prediction_output:
    prediction = [str(x) for x in prediction]
    f.write(','.join(prediction))
    f.write('\n')
f.close()