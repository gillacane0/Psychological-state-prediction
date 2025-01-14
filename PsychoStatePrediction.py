import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from CognitiveLoadPrediction import t_hat_test

df = pd.read_csv("psychological_state_dataset.csv")

#EDA
print(df.info())
print("NULL VALUES: \n",df.isnull().sum())
print("DUPLICATED VALUES\n",df.duplicated().sum())

df = df.drop(['ID'], axis=1)
df = df.drop(['Time'], axis=1)

def extract_values(row):
    return ast.literal_eval(row)

def split_values(row):
    a, b = row.split("/")
    return int(a), int(b)

df['Bands'] = df['EEG Power Bands'].apply(extract_values)
df['EEG_Delta'] = df['Bands'].apply(lambda x: x[0])
df['EEG_Alpha'] = df['Bands'].apply(lambda x: x[1])
df['EEG_Beta'] = df['Bands'].apply(lambda x: x[2])


df['Pressure_Systolic'] = df['Blood Pressure (mmHg)'].apply(lambda x: split_values(x)[0])
df['Pressure_Diastolic'] = df['Blood Pressure (mmHg)'].apply(lambda x: split_values(x)[1])

df.drop(['Blood Pressure (mmHg)','Bands','EEG Power Bands'],axis=1,inplace=True)

df.replace({'Psychological State': 'Stressed'}, 0, inplace=True)
df.replace({'Psychological State': 'Relaxed'}, 1, inplace=True)
df.replace({'Psychological State': 'Focused'}, 2, inplace=True)
df.replace({'Psychological State': 'Anxious'}, 3, inplace=True)
pd.set_option('future.no_silent_downcasting', True)

t = df['Psychological State'].values
X = df.drop(['Psychological State'],axis=1)

X = pd.get_dummies(X)


print(X.info())

#CORRELATION MATRIX
corr_Matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 11))
sns.heatmap(corr_Matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
plt.title('Correlation matrix', fontsize=14, pad=20)
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#TRAIN TEST DEV SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2,
                                                    random_state=42)
X_train, X_dev, t_train, t_dev = train_test_split(X_train, t_train,
                                                    test_size=0.1, random_state=42)

#NORMALIZATION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_dev = sc.transform(X_dev)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#commented out to avoid running the code again
""""
# Log regression
logreg = LogisticRegression()
param_grid_lr = {
    'C': [1, 10, 100, 1000,10000]
}
clf_lr = GridSearchCV(estimator=logreg,
                      param_grid=param_grid_lr,
                      scoring='f1_weighted',
                      n_jobs=-1,
                      cv=8,
                      verbose=10,
                      refit=True)
clf_lr.fit(X_train, t_train)

#SVM
svc = SVC()
param_grid_svc = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [1, 10, 100, 1000],
    'gamma': [1e-3, 1e-4,1e-5],
}
clf_svm = GridSearchCV(estimator=svc,
                      param_grid=param_grid_svc,
                      scoring='f1_weighted',
                      n_jobs=-1,
                      cv=8,
                      verbose=10,
                      refit=True)
clf_svm.fit(X_train, t_train)

#NN
nn = MLPClassifier()
param_grid_nn = {
    'hidden_layer_sizes': [(100, 100), (200, 100),(200,200),(300,150)],
    'alpha': [1e-2, 1e-3, 1e-4],
    'activation': ['relu','tanh'],
}
clf_nn = GridSearchCV(estimator=nn,
                      param_grid=param_grid_nn,
                      scoring='f1_weighted',
                      n_jobs=-1,
                      cv=8,
                      verbose=10,
                      refit=True)
clf_nn.fit(X_train, t_train)

# Best params
print("Best hyperparameters Log Reg: ", clf_lr.best_params_)
print("Best hyperparameters SVM: ", clf_svm.best_params_)
print("Best hyperparameters NN: ", clf_nn.best_params_)

# Test on development set
t_hat_dev_lr = clf_lr.predict(X_dev)
t_hat_dev_svm = clf_svm.predict(X_dev)
t_hat_dev_nn = clf_nn.predict(X_dev)
print("Classification report for Log reg on Dev set: ",
      classification_report(t_dev, t_hat_dev_lr))
print("Classification report for SVM on Dev set: ",
      classification_report(t_dev, t_hat_dev_svm))
print("Classification report for NN on Dev set: ",
      classification_report(t_dev, t_hat_dev_nn))


t_dev = tf.cast(t_dev, dtype=tf.int64)
t_hat_dev_nn = tf.cast(t_hat_dev_nn, dtype=tf.float32)
t_hat_dev_svm = tf.cast(t_hat_dev_svm, dtype=tf.float32)

lossNN = tf.keras.losses.categorical_crossentropy(t_dev,t_hat_dev_svm)
lossSVC = tf.keras.losses.categorical_crossentropy(t_dev,t_hat_dev_nn)

print("VALUE OF CROSS ENTROPY LOSS FOR DEV SET FOR SVM: ",lossSVC.numpy())
print("VALUE OF CROSS ENTROPY LOSS FOR DEV SET FOR NN: ",lossNN.numpy())

"""

# Merge Train and development
X_train_final = np.vstack((X_train, X_dev))
t_train_final = np.hstack((t_train, t_dev))

#FINAL MODELS
nn = MLPClassifier(activation='relu', alpha=0.0001,hidden_layer_sizes=(100,100),solver='adam')
nn.fit(X_train_final,t_train_final)

t_hat_test = nn.predict(X_test)

print("DIO CANE\n\n")
print("CLASSIFICATION REPORT FOR NN ",classification_report(t_test,t_hat_test))

print("SERPENTE IL SIGNORE")

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(t_test,t_hat_test)
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
plt.colorbar(cax)

class_names = ['Stressed', 'Relaxed', 'Focused', 'Anxious']
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, f'{value}', ha='center', va='center', color='black')

plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()