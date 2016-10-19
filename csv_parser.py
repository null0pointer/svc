def parse_training_csv(csv):
    # Set up arrays for features and targets
    features = []
    targets = []

    lines = lines_for_csv(csv)

    # Get the number of columns
    columns = len(lines[0].split(','))

    # Remove the title row from the lines
    lines = lines[1:]

    # Parse the lines into arrays and add to features and targets
    for line in lines:
        values = line.split(',')
        if len(values) == columns:
            targets.append(float(values[-1]))
            features.append([float(x) for x in values[:-1]])
            
    return (features, targets)
    
def parse_prediction_csv(csv):
    # Set up arrays for predictions
    ids = []
    features = []
    
    lines = lines_for_csv(csv)
    
    # Get the number of columns
    columns = len(lines[0].split(','))

    # Remove the title row from the lines
    lines = lines[1:]

    # Parse the lines into arrays and add to features and ids
    for line in lines:
        values = line.split(',')
        if len(values) == columns:
            ids.append(str(values[0]))
            features.append([float(x) for x in values[1:]])
            
    return (ids, features)
    
def write_lines_to_csv(csv, lines):
    # Write to CSV
    f = open(csv, 'w')
    for line in lines:
        line = [str(x) for x in line]
        f.write(','.join(line))
        f.write('\n')
    f.close()
    
def lines_for_csv(csv):
    # Open the training data CSV
    f = open(csv, 'r')

    # Split the CSV into lines
    lines = f.read().split('\n')
    f.close()
    
    return lines