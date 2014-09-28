import statsmodels.api as sm
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv('train.csv', header = 0)

##Data Queries

#print df.Age[0:10]
#print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
#df['Age'].hist()
#pl.show()

#Create column with integer representation of Sex
df['Gender'] = 4
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Fill in missing Ages with typical median values based on gender and social status in a new column
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
                              
df['AgeFill'] = df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

#print df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

#Create column with integer values for Embarked, 2 missing values filled with most frequently occuring values
df['PortEmbarked'] = df['Embarked']
mode_embarked = df['Embarked'].mode()
df.loc[ (df.Embarked.isnull()),'PortEmbarked'] = mode_embarked[0]
df['PortEmbarked'] = df['PortEmbarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
#print df['PortEmbarked'].head(10)

#Combine number of parents/children and number of siblings/spouses
df['FamilySize'] = df['SibSp'] + df['Parch']

#df['Age*Class'] = df.AgeFill * df.Pclass

#Drop columns that have string objects or are repeated
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','SibSp','Parch'], axis=1) 

#Drop columns that we assume will not matter 
df = df.drop(['PassengerId'], axis=1);

train_data = df.values
#df.info()
#print train_data[0]

train_cols = train_data[:,1:]
#print train_cols[0]

#Create logistic regression model
logit = sm.Logit(train_data[:,0], train_cols)
result = logit.fit()

####################################################
df_test = pd.read_csv('test.csv', header = 0)
#df_test.info()

#Create column with integer representation of Sex
df_test['Gender'] = 4
df_test['Gender'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Fill in missing Ages with typical median values based on gender and social status in a new column
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df_test[(df_test['Gender'] == i) & \
                              (df_test['Pclass'] == j+1)]['Age'].dropna().median()
                              
df_test['AgeFill'] = df_test['Age']
for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[ (df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

#Create integer column (PortEmbarked) for Embarked           
df_test['PortEmbarked'] = 0
df_test['PortEmbarked'] = df_test['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

#Combine number of parents/children and number of siblings/spouses
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

#Add value of missing Fare as median value of Fare
df_test.loc[ (df_test.Fare.isnull()),'Fare'] = df_test['Fare'].dropna().median()

#Drop insignificant columns
df_test = df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','SibSp','Parch'], axis=1) 

#Test data using model
test_data = df_test.values
independent_variables = test_data[:,1:]
predictions = []
predictions = result.predict(independent_variables)

#Use 0.5 as a threshold to determine survival or death
predictions[predictions < 0.5] = 0
predictions[predictions >= 0.5] = 1

#Create results array with data type int as required by Kaggle
result_arr = test_data[:,0]
print len(result_arr)
print len(predictions)
result_arr = np.append(np.vstack(result_arr), np.vstack(predictions), 1)
result_arr = result_arr.astype(int)
#print result_arr.dtype
#print result_arr

#Write results to csv file
import csv
with open('titanic_results.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerow(['PassengerId','Survived'])
    a.writerows(result_arr)
