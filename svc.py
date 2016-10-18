from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]

classifier = svm.SVC(probability=True, verbose=True)
classifier.fit(X, y)

print(classifier.predict([[2, 2]]))

print(classifier.predict_proba([[2, 2]]))