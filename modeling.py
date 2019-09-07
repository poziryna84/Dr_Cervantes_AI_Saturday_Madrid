# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:53:58 2019

@author: pozir
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

'''
                           Random Forest estimation
'''
#==============================================================================
# Importing treated data
#==============================================================================

df_trans = pd.read_pickle('data/trans.pickle')
df_id = pd.read_pickle('data/id.pickle')

df = pd.merge(df_trans, 
              df_id,
              how = 'left', 
              on = 'TransactionID').drop(
                      'TransactionID', axis = 1)

del df_trans, df_id 

df_dummies = pd.get_dummies(df[['card4', 'card6',
             'ProductCD', 'first_dg_card1', 'first_dg_card2', 'first_dg_card3', 'first_dg_card5', 'second_dg_card1',
             'second_dg_card2', 'second_dg_card3', 'second_dg_card5','id_12', 'id_15',
             'id_16','id_23','id_27','id_28','id_29', 'id_30', 'id_32', 'id_34',
             'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType']])

df = pd.concat([df, df_dummies], axis=1).drop(columns = ['card4', 'card6',
             'ProductCD', 'first_dg_card1', 'first_dg_card2', 'first_dg_card3', 'first_dg_card5', 'second_dg_card1',
             'second_dg_card2', 'second_dg_card3', 'second_dg_card5','id_12', 'id_15',
             'id_16','id_23','id_27','id_28','id_29', 'id_30', 'id_32', 'id_34',
             'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType'])
del df_dummies

X = df.drop(['isFraud', 'id_34_match_status:-1', 'card6_charge card'], axis = 1)
y = df['isFraud']

del df

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, 
        stratify = y,
        random_state=42)
del X, y

clf = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=None,
                             random_state=123, class_weight=None)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(f1_score(y_test, predictions, average = None))
print(classification_report(y_test, predictions))
print(roc_auc_score(y_test, predictions))
importances = clf.feature_importances_

feature_importance = pd.DataFrame({
        'variables' : list(X_train.columns),
        'score' : list(importances)
        }).sort_values(by=['score'],ascending=False)

