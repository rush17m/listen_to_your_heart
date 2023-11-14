import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

from lime import lime_tabular

'''
Using Breast cancer data to predict the cancer probability
In this case, the model used is called Extra Tree Regressor model
It choses features and splits at random. 

It is an ensemble supervised machine learning methid that ises
decision trees
'''



data = load_breast_cancer()
X = data['data']
y = data['target']
features = data['feature_names']

# Uncomment if need to see the data in detail
# print('Here is the data size: ', data.data.shape)
# print('Here are the feature names: ', data.feature_names)
# print('Here are the target names: ', data.target_names)
# print('Here is the dataset description: ', data.DESCR)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=20)

df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
df.columns = np.concatenate((features, np.array(['label'])))

print('Shape of data = ', df.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

df.head()

reg = ExtraTreesRegressor(random_state=30)

reg.fit(X_train, y_train)

print('Score for the model on test set = ', reg.score(X_test, y_test))

explainer_lime = lime_tabular.LimeTabularExplainer(X_train, feature_names=features, verbose=True, mode='regression')

#running the lime explainer for specific test_vector and num_of_features
num_of_top_features = 5

file =open('extra_tree_regressor_lime_explanation.html','w', encoding="utf-8")
for _ in range(15):
    exp = explainer_lime.explain_instance(X_test[random.randrange(len(X_test))], reg.predict, num_features=num_of_top_features)
    print(file.write(exp.as_html()))