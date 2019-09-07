# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:53:58 2019

@author: pozir
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

selection = ['length_card1', 'length_card2',  'length_card5', 'TransactionAmt', 'card4', 'card6',
             'ProductCD', 'first_dg_card1', 'first_dg_card2', 'first_dg_card3', 'first_dg_card5', 'second_dg_card1',
             'second_dg_card2', 'second_dg_card3', 'second_dg_card5', 'isFraud', 'addr1_bin', 'addr2_bin',
             'P_emaildomain_bin', 'R_emaildomain_bin', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
             'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
             'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'P_valid_email', 'R_valid_email']

#selection = selection + vs_list
df = train_trans[selection]
pca = train_trans[vs_list]
pca_data = StandardScaler().fit_transform(pca)

pca = PCA(n_components= 150)
pca_data = pca.fit_transform(pca_data)

pca_data = pd.DataFrame(data=pca_data,    # 1st column as index
            columns=['PCA_' + str(i) for i in range(1,151)])
del train_trans
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance') # 135

#distortions = []
#for i in range(1, 20):
#    km = KMeans(
#        n_clusters=i, init='random',
#        n_init=10, max_iter=300,
#        tol=1e-04, random_state=0
#    )
#    km.fit(pca_data)
#    distortions.append(km.inertia_)
#
## plot
#plt.plot(range(1, 20), distortions, marker='o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()

km =  KMeans(
        n_clusters=5, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

pca5data = pd.DataFrame(km.fit_predict(pca_data))
pca5data.index = pca_data.index


df = pd.concat([df, pca5data], axis = 1)

#df = df.rename(columns={'0': 'cluster'})
#
#for i in df:
#    print(i)

del pca_data, pca5data
transaction_col =list( train_trans.columns)
#df.dropna(inplace = True)

df_dummies = pd.get_dummies(df[['card4', 'card6',
             'ProductCD', 'first_dg_card1', 'first_dg_card2', 'first_dg_card3', 'first_dg_card5', 'second_dg_card1',
             'second_dg_card2', 'second_dg_card3', 'second_dg_card5']])

df = pd.concat([df, df_dummies], axis=1).drop(columns = ['card4', 'card6',
             'ProductCD', 'first_dg_card1', 'first_dg_card2', 'first_dg_card3', 'first_dg_card5', 'second_dg_card1',
             'second_dg_card2', 'second_dg_card3', 'second_dg_card5'])
del df_dummies

X = df.drop(['isFraud'], axis = 1)
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

