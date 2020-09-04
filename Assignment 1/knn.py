import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
print(X_train)
print(Y_train)

# define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')

# fit model
classifier.fit(X_train, Y_train)

# predict the test set results
Y_pred = classifier.predict(X_val)

# evaluate model
cm = confusion_matrix(Y_val, Y_pred)
print (cm)

print("F1 score: ", f1_score(Y_val, Y_pred))
print("Accuracy: ", accuracy_score(Y_val, Y_pred))
