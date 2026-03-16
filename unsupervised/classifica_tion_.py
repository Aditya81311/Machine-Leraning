# Loading Modules 
from sklearn import datasets  #importing dataset
from sklearn.neighbors import KNeighborsClassifier #importing Classifier
ds = datasets.load_iris() #loading dateset from datasets

#Defining Features and Labels 
features = ds.data
labels = ds.target

# print(features[0],labels[0])

# Defing Classifier
clf = KNeighborsClassifier()

#Fitting Data
clf.fit(features,labels)

#Predecting 
pre = clf.predict([[5.1, 3.5, 1.4, 0.2]])

print(pre)