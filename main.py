# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:31:50 2019

@author: pozir
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle


train_trans = pd.read_csv('data/train_transaction.csv')

#==============================================================================
# isFraud
#==============================================================================

#train_trans.isFraud.value_counts(normalize = True)
#train_trans.isFraud.isna().sum()
#
#train_trans.hist(column='isFraud')
#plt.title("isFraud")
#plt.show()
#
##==============================================================================
## TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
##==============================================================================
#
#train_trans.TransactionDT.describe()
#train_trans.TransactionDT.isna().sum() # 0
#
#train_trans.hist(column='TransactionDT')
#plt.title("TransactionDT Total")
#plt.show()
#
#train_trans[train_trans.isFraud == 0].hist(column='TransactionDT')
#plt.title("TransactionDT Non Fraud")
#plt.show()
#
#train_trans[train_trans.isFraud == 1].hist(column='TransactionDT')
#plt.title("TransactionDT Fraud")
#plt.show()

#
#train_trans.days_ago = round(train_trans.TransactionDT/86400)
#train_trans.days_ago.isna().sum()
#train_trans.days_ago = train_trans.days_ago.astype(int)
#trans_delta = train_trans.days_ago.value_counts().to_frame(
#        ).reset_index().rename(columns ={'index' : 'days',
#                                               'TransactionDT' : 'count_trans'})
#
#train_trans = pd.merge(train_trans, trans_delta, 
#                       left_on = 'days_ago',
#                       right_on = 'days',
#                       how = 'left')
##==============================================================================
# TransactionAmt: transaction payment amount in USD
#==============================================================================

train_trans.TransactionAmt.describe()
train_trans.TransactionAmt.isna().sum() # 0

train_trans.TransactionAmt[train_trans.isFraud == 0].describe()
train_trans.TransactionAmt[train_trans.isFraud == 1].describe()

train_trans.hist(column='TransactionAmt')
plt.title("TransactionAmt Total")
plt.show()

train_trans[train_trans.isFraud == 0].hist(column='TransactionAmt')
plt.title("TransactionAmt Non Fraud")
plt.show()

train_trans[train_trans.isFraud == 1].hist(column='TransactionAmt')
plt.title("TransactionAmt Fraud")
plt.show()

#==============================================================================
# card1 - card6: payment card information, such as card type, card category, issue bank, country, etc
#==============================================================================

train_trans.card4.value_counts()

train_trans['length_card1'] = train_trans['card1'].apply(lambda x: len(str(x)) if type(x) == int else 0)

train_trans['length_card2'] = np.where(train_trans['card2'].notna(),1,0)

train_trans['length_card3'] = np.where(train_trans['card3'].notna(),1,0)

train_trans['length_card5'] = np.where(train_trans['card5'].notna(),1,0)

#==============================================================================
# First and second digits
#==============================================================================
train_trans['first_dg_card1'] = train_trans['card1'].apply(lambda x: str(x)[0] if type(x) == int else 'none')
train_trans['second_dg_card1'] = train_trans['card1'].apply(lambda x: str(x)[1] if type(x) == int else 'none')

train_trans['first_dg_card2'] = train_trans['card2'].apply(lambda x: str(x)[0] if x > 0.0 else 'none')
train_trans['second_dg_card2'] = train_trans['card2'].apply(lambda x: str(x)[1] if x > 0.0 else 'none')

train_trans['first_dg_card3'] = train_trans['card3'].apply(lambda x: str(x)[0] if x > 0.0 else 'none')
train_trans['second_dg_card3'] = train_trans['card3'].apply(lambda x: str(x)[1] if x > 0.0 else 'none')

train_trans['first_dg_card5'] = train_trans['card5'].apply(lambda x: str(x)[0] if x > 0.0 else 'none')
train_trans['second_dg_card5'] = train_trans['card5'].apply(lambda x: str(x)[1] if x > 0.0 else 'none')

train_trans['card4'].fillna('none', inplace = True)
train_trans['card6'].fillna('none', inplace = True)

#==============================================================================
# addr1 and addr2
#==============================================================================

train_trans['addr1_bin'] = np.where(train_trans['addr1'].notna(),1,0)
train_trans['addr2_bin'] = np.where(train_trans['addr2'].notna(),1,0)

#==============================================================================
#  P_emaildomain and R_emaildomain
#==============================================================================

train_trans['P_emaildomain_bin'] = np.where(train_trans['P_emaildomain'].notna(),1,0)
train_trans['R_emaildomain_bin'] = np.where(train_trans['R_emaildomain'].notna(),1,0)

from urllib.request import urlopen

target_url = 'https://gist.githubusercontent.com/tbrianjones/5992856/raw/93213efb652749e226e69884d6c048e595c1280a/free_email_provider_domains.txt'
data = urlopen(target_url) 

valid_email_list = []

for line in data: # files are iterable
    email = line.decode("utf-8")
    email = email.replace('\n', '')
    valid_email_list.append(email)

train_trans['P_valid_email'] = np.where(train_trans['P_emaildomain'].isin(valid_email_list),1,0)   
train_trans['R_valid_email'] = np.where(train_trans['R_emaildomain'].isin(valid_email_list),1,0)   

#==============================================================================
# D1-D15
#==============================================================================
ds_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
    
def fillna(df, columns_list):
    for i in df[columns_list]:
        df[i] = df[i].fillna(-1)
    return df
            
fillna(df = train_trans, columns_list = ds_list)
    
#==============================================================================
# V1-V339    
#==============================================================================

vs_list = []

for i in train_trans:
    if i.startswith('V'):
        vs_list.append(i)
        
fillna(df = train_trans, columns_list = vs_list)

#==============================================================================
# PCA & Clustering
#==============================================================================

selection = ['TransactionID','length_card1', 'length_card2',  'length_card5', 'TransactionAmt', 'card4', 'card6',
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

km =  KMeans(
        n_clusters=5, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

pca5data = pd.DataFrame(km.fit_predict(pca_data))
pca5data.index = pca_data.index


df = pd.concat([df, pca5data], axis = 1)
df.to_pickle('data/trans.pickle')

del pca_data, pca5data, df