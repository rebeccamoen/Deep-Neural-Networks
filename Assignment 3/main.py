import csv, math
import sys
import pandas as pd

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

data = csv.reader(open("breast-cancer.data", "r"))
print(data)
alldata = []

print("recrusion: ", sys.getrecursionlimit())
sys.setrecursionlimit(10000)

"""
   1. Class: no-recurrence-events, recurrence-events
   2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
   3. menopause: lt40, ge40, premeno.
   4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44,
                  45-49, 50-54, 55-59.
   5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26,
                 27-29, 30-32, 33-35, 36-39.
   6. node-caps: yes, no.
   7. deg-malig: 1, 2, 3.
   8. breast: left, right.
   9. breast-quad: left-up, left-low, right-up,	right-low, central.
  10. irradiat:	yes, no.
"""

for d in data:
    alldata.append([1 if d[0] == "recurrence-events" else 0,d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],])
    #alldata.append([1 if d[0] == "recurrence-events" else 0,d[1], d[2], d[9]])
    data_text=["Reoccurence", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
    # Reoccurence, age, menopause, tumor-size, inv-nodes(?), node-caps, deg-malig(?), breast (left/right), breast-quad, irradiat

print(alldata)
alldata = shuffle(alldata)

trainingdata = alldata[int(len(alldata) / 2):]
verificationdata = alldata[:int(len(alldata) / 2)]


def entropy(oneclass):
    no_recurr = len([i for i in oneclass if i[0] == 0])
    recurr = len([i for i in oneclass if i[0] == 1])
    total = no_recurr + recurr
    if (min(no_recurr, recurr) == 0):
        return 0
    entropy = - (no_recurr / total) * math.log(no_recurr / total, 2) - (recurr / total) * math.log(recurr / total, 2)
    return entropy


print(entropy(trainingdata))


def split(data, attribute, remove=False):
    # import pdb;pdb.set_trace()
    retvals = {}
    allattributes = set([i[attribute] for i in data])
    for d in data:
        c = d[attribute]
        aList = retvals.get(c, [])
        if (remove):
            d.pop(attribute)
        aList.append(d)
        retvals[c] = aList
    return retvals


def gain(oneclass, attribute):
    d = [(entropy(i), len(i)) for i in split(oneclass, attribute).values()]
    nAll = sum(i[1] for i in d)
    gain = sum([(i[0] * i[1]) / nAll for i in d])
    return gain


def getHighestGain(oneclass):
    classes = [i for i in range(1, len(oneclass[0]))]
    entropies = [gain(oneclass, c) for c in classes]
    return entropies.index(min(entropies)) + 1


print(getHighestGain(trainingdata))


def isPure(oneclass):
    classes = [i for i in range(1, len(oneclass[0]))]

    for c in classes:
        if (len(set([i[c] for i in oneclass])) > 1):
            import pdb;
#            pdb.set_trace()
            return False
    return True


def isEmpty(oneclass):
    return len(oneclass[0]) <= 1


def mostCommon(oneclass):
    lst = [i[0] for i in oneclass]
    return max(set(lst), key=lst.count)


actualClassifier = "def classify(data):"
prevSplit = -1
totleafes = 0
def buildTree(oneclass, spaces="    "):
    global actualClassifier
    global prevSplit
    if (isEmpty(oneclass) or isPure(oneclass) or prevSplit==getHighestGain(oneclass)):
        print(spaces, "then", mostCommon(oneclass))
        # print(spaces,"#confidence",confidence(oneclass))
        actualClassifier += "\n" + spaces + "return (" + str(mostCommon(oneclass)) + ")"
        return
    highest = getHighestGain(oneclass)

    d = split(oneclass, highest)
    prevSplit = highest
    for key, value in d.items():
        print(spaces, "if", key, "   #",data_text[highest])
        global totleafes
        totleafes += 1
        actualClassifier += "\n" + spaces + "if(data[" + str(highest) + "]==\"" + str(key) + "\"):"

        if(len(spaces) <= 20):
            buildTree(value, spaces + "   ")
        else:
            return

buildTree(trainingdata)
print(actualClassifier)

exec(actualClassifier)
print(classify(verificationdata[0]), verificationdata[0])

correct, wrong = 0, 0
for data in verificationdata:
    if ((data[0]) == (classify(data))):
        correct += 1
    else:
        wrong += 1
print("Correct classifications", correct)
print("Wrong classifications", wrong)
print("Accuracy", (correct / (correct + wrong)))
print("Leafes: ", totleafes)


###############################################
#From this line we generate random forest, before this line is decision tree

dataset = pd.read_csv("breast-cancer.data")


dataset.head()
dataset = shuffle(dataset)



#dataset = np.array(columnTransformer.fit_transform(dataset), dtype=np.str)

X = dataset.iloc[:, 1:9].values
y = dataset.iloc[:,0].values



#Have to remove all strings into float or int. Most of the data is in string form.
lencodX = []
lencodY = LabelEncoder()

for i in range(len(X[0])):
    lencode = LabelEncoder()
    X[:, i] = lencode.fit_transform(X[:,i])
    lencodX.insert(i,lencode)



for i in range(len(y)):
    if y[i] == "recurrence-events":
        y[i] = 1
    else:
        y[i] = 0

y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=10, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test, y_pred))
