

import pandas as pd
import numpy as np

df=pd.read_csv("winequalityN.csv")

df

df.info()

df.nunique()

import matplotlib.pyplot as plt

plt.pie(df.type.value_counts().values,labels=df.type.value_counts().index,shadow=True,autopct="%1.2f%%")

df.isna().sum()

plt.hist(df['fixed acidity'])

np.mean(df['fixed acidity'])

plt.hist(df['volatile acidity'])

plt.hist(df['citric acid'])

plt.hist(df['residual sugar'])

plt.hist(df['chlorides'])

plt.hist(df['density'])

plt.hist(df['pH'])

plt.hist(df['sulphates'])

plt.hist(df['alcohol'])

plt.pie(df.quality.value_counts().values,labels=df.quality.value_counts().index,shadow=True,autopct="%1.2f%%")



import seaborn as sns
sns.boxplot(df['fixed acidity'])

print(df['fixed acidity'].quantile(0.10))
print(df['fixed acidity'].quantile(0.90))

df['fixed acidity']=np.where(df['fixed acidity']<6.0,6.0,df['fixed acidity'])
df['fixed acidity']=np.where(df['fixed acidity']>8.8,8.8,df['fixed acidity'])

sns.boxplot(df['fixed acidity'])

sns.boxplot(df['volatile acidity'])

print(df['volatile acidity'].quantile(0.10))
print(df['volatile acidity'].quantile(0.90))

df['volatile acidity']=np.where(df['volatile acidity']<0.18,0.18,df['volatile acidity'])
df['volatile acidity']=np.where(df['volatile acidity']>0.59,0.59,df['volatile acidity'])

sns.boxplot(df['volatile acidity'])

sns.boxplot(df['citric acid'])

print(df['citric acid'].quantile(0.10))
print(df['citric acid'].quantile(0.90))

df['citric acid']=np.where(df['citric acid']<0.14,0.14,df['citric acid'])
df['citric acid']=np.where(df['citric acid']>0.49,0.49,df['citric acid'])

sns.boxplot(df['citric acid'])

sns.boxplot(df['residual sugar'])

print(df['residual sugar'].quantile(0.10))
print(df['residual sugar'].quantile(0.90))

df['residual sugar']=np.where(df['residual sugar']<1.3,1.3,df['residual sugar'])
df['residual sugar']=np.where(df['residual sugar']>13.0,13,df['residual sugar'])

sns.boxplot(df['residual sugar'])



sns.boxplot(df['chlorides'])

print(df['chlorides'].quantile(0.10))
print(df['chlorides'].quantile(0.90))

df['chlorides']=np.where(df['chlorides']<0.031,0.031,df['chlorides'])
df['chlorides']=np.where(df['chlorides']>0.086,0.086,df['chlorides'])

sns.boxplot(df['chlorides'])

print(df['free sulfur dioxide'].quantile(0.10))
print(df['free sulfur dioxide'].quantile(0.90))

df['free sulfur dioxide']=np.where(df['free sulfur dioxide']<9.0,9.0,df['free sulfur dioxide'])
df['free sulfur dioxide']=np.where(df['free sulfur dioxide']>54.0,54.0,df['free sulfur dioxide'])

sns.boxplot(df['free sulfur dioxide'])

sns.boxplot(df['total sulfur dioxide'])

print(df['total sulfur dioxide'].quantile(0.10))
print(df['total sulfur dioxide'].quantile(0.90))

df['total sulfur dioxide']=np.where(df['total sulfur dioxide']<30.0,30.0,df['total sulfur dioxide'])
df['total sulfur dioxide']=np.where(df['total sulfur dioxide']>188.0,188.0,df['total sulfur dioxide'])

sns.boxplot(df['total sulfur dioxide'])

sns.boxplot(df['density'])

print(df['density'].quantile(0.10))
print(df['density'].quantile(0.95))

df['density']=np.where(df['density']<0.99068,0.99068,df['density'])
df['density']=np.where(df['density']>0.999395,0.999395,df['density'])

sns.boxplot(df['density'])

sns.boxplot(df['pH'])

print(df['pH'].quantile(0.05))
print(df['pH'].quantile(0.95))

df['pH']=np.where(df['pH']<2.97,2.97,df['pH'])
df['pH']=np.where(df['pH']>3.5,3.5,df['pH'])

sns.boxplot(df['pH'])

sns.boxplot(df['sulphates'])

print(df['sulphates'].quantile(0.05))
print(df['sulphates'].quantile(0.95))

df['sulphates']=np.where(df['sulphates']<0.35,0.35,df['sulphates'])
df['sulphates']=np.where(df['sulphates']>0.79,0.79,df['sulphates'])

sns.boxplot(df['sulphates'])

df.isna().sum()

sns.heatmap(df.corr(),annot=True,cmap='RdBu')
plt.show()

np.mean(df['fixed acidity'])

df['fixed acidity'].fillna(np.mean(df['fixed acidity']),inplace =True)

df['volatile acidity'].median()

df['volatile acidity'].fillna(df['volatile acidity'].median(),inplace =True)

plt.hist(df['citric acid'])

df['citric acid'].fillna(np.mean(df['citric acid']),inplace =True)

plt.hist(df['residual sugar'])

df['residual sugar'].fillna(df['residual sugar'].median(),inplace =True)

plt.hist(df['chlorides'])

df['chlorides'].fillna(df['chlorides'].median(),inplace =True)

plt.hist(df['density'])

print(df['density'].median())
np.mean(df['density'])

df['density'].fillna(df['density'].median(),inplace =True)

plt.hist(df['pH'])

df['pH'].fillna(np.mean(df['pH']),inplace =True)

plt.hist(df['sulphates'])

df['sulphates'].fillna(df['sulphates'].median(),inplace =True)

df.isna().sum()

df.info()

df.nunique()

df['type']=df.apply(lambda x:1 if x['type']=='White' else 0,axis=1)

df.info()

x=df.iloc[:,:-1]
x.head()

y=df.iloc[:,-1]
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

numeric_processor = Pipeline(
    steps=[
        ('imputation_mean', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler())
    ]
)

numeric_processor

from sklearn.preprocessing import OneHotEncoder

categorical_processor = Pipeline(
    steps=[
        ('imputation_constant', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

categorical_processor

## combine processing techniques
from sklearn.compose import ColumnTransformer

df.info()

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', categorical_processor, ['type']),
        ('numeric', numeric_processor, [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ])
    ]
)

preprocessor

from sklearn.pipeline import make_pipeline

pipe= make_pipeline(preprocessor,LogisticRegression())

pipe

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

from sklearn import metrics
cnf_mx = metrics.confusion_matrix(y_test, y_pred)
cnf_mx

sns.heatmap(cnf_mx,annot=True,annot_kws = {'size':15},fmt=".0f")
plt.xlabel("Predict")
plt.ylabel("Actual")

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# Creating a pipeline with OneVsRestClassifier and logistic regression
pipe_multiclass = make_pipeline(
    preprocessor,
    OneVsRestClassifier(LogisticRegression())
)

# Fitting the model using pipeline
pipe_multiclass.fit(X_train, y_train)

# Predictions on test set
y_proba_multiclass = pipe_multiclass.predict_proba(X_test)

# Computing ROC-AUC for each class
roc_auc_scores = []
plt.figure(figsize=(10, 6))

for i in range(len(pipe_multiclass.classes_)):
    fpr_i, tpr_i, _ = roc_curve(y_test == pipe_multiclass.classes_[i], y_proba_multiclass[:, i])
    roc_auc_i = auc(fpr_i, tpr_i)
    roc_auc_scores.append(roc_auc_i)

    # Plotting ROC Curve for each class
    plt.plot(fpr_i, tpr_i, lw=2, label=f'ROC curve (area = {roc_auc_i:.2f}) for class {pipe_multiclass.classes_[i]}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for Multiclass Classification')
plt.legend(loc="lower right")
plt.show()


from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

categorical_features = ['type']

# Defining the classifier without PCA
pipe_no_pca_1 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Defining the classifier with PCA
pipe_pca_1 = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Evaluating the models using cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Model without PCA
scores_no_pca = cross_val_score(pipe_no_pca_1, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f'Accuracy without PCA: {np.mean(scores_no_pca):.2f}')

# Model with PCA
scores_pca = cross_val_score(pipe_pca_1, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f'Accuracy with PCA: {np.mean(scores_pca):.2f}')

import pickle

with open('wine_quality_model.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)

