from sklearn.datasets import load_iris #iris datatset
from sklearn.ensemble import RandomForestClassifier #rfc library
import pandas as pd #helps creating a dataframe which looks like excel spreadsheet
import numpy as np

np.random.seed(0) #setting random seed
iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.head())

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) #adding new column for species name
print(df.head())

df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.75
print(df.head())

train, test = df[df['is_train']==True], df[df['is_train']==False]
print(len(train))
print(len(test))
print(len(df))

features = df.columns[:4] #creating list of feature's column names
print(features)

y = pd.factorize(train['species'])[0] #converting each species name into digits
print(y)

clf = RandomForestClassifier(n_jobs = 2, random_state=0) #creating rfc
clf.fit(train[features],y) #training code

clf.predict(test[features]) #applying trained classifier to test

clf.predict_proba(test[features])[0:10] #viewing prediction probabilities of first ten data

preds = iris.target_names[clf.predict(test[features])]#marking the species names for the predicted data
print(preds[0:5])

test['species'].head() #this is actual data
pd.crosstab(test['species'],preds,rownames=['Actual Species'],colnames=['Predicted Species'])
    #rownames and colnames are actual parameters in crosstab func defn
    #crosstab() creates a chart out of two sets of data.. first rows and then columns
    #totsl number of correct predictions =30 and inaccurate predictions = 2
    #model accuracy =93%
