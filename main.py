# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:31:50 2019

@author: pozir
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_ident = pd.read_csv('data/train_identity.csv')
train_trans = pd.read_csv('data/train_transaction.csv')


# Not all transactions have corresponding identity information,
# so let's get rid of unnecessary transaction data:
    
#train_trans = train_trans[train_trans.TransactionID.isin(train_ident.TransactionID)]

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


train_trans.days_ago = round(train_trans.TransactionDT/86400)
train_trans.days_ago.isna().sum()
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
# ProductCD: product code, the product for each transaction
#==============================================================================

train_trans.ProductCD.value_counts(normalize = True)
train_trans.ProductCD.isna().sum() # 0

train_trans['ProductCD'].value_counts().plot(kind='bar')

train_trans.ProductCD[train_trans.isFraud == 1].value_counts(normalize = True)
train_trans.ProductCD[train_trans.isFraud == 0].value_counts(normalize = True)

train_trans.isFraud[train_trans.ProductCD == 'C'].value_counts(normalize = True)
train_trans.isFraud[train_trans.ProductCD == 'R'].value_counts(normalize = True)
train_trans.isFraud[train_trans.ProductCD == 'H'].value_counts(normalize = True)
train_trans.isFraud[train_trans.ProductCD == 'S'].value_counts(normalize = True)


train_trans.TransactionAmt[train_trans.ProductCD == 'C'].describe()
train_trans.TransactionAmt[train_trans.ProductCD == 'R'].describe()
train_trans.TransactionAmt[train_trans.ProductCD == 'H'].describe()
train_trans.TransactionAmt[train_trans.ProductCD == 'S'].describe()

#==============================================================================
# card1 - card6: payment card information, such as card type, card category, issue bank, country, etc
#==============================================================================

train_trans.card4.value_counts()

train_trans['length_card1'] = train_trans['card1'].apply(lambda x: len(str(x)) if type(x) == int else 0)

#train_trans['length_card2'] = train_trans['card2'][train_trans['card2'].notna()].apply(lambda x: x = 1)
#train_trans['length_card2'].fillna(0, inplace = True)

train_trans['length_card2'] = np.where(train_trans['card2'].notna(),1,0)

#train_trans['length_card3'] = train_trans['card3'][train_trans['card3'].notna()].apply(lambda x: x = 1)
#train_trans['length_card3'].fillna(0, inplace = True)

train_trans['length_card3'] = np.where(train_trans['card3'].notna(),1,0)

#train_trans['length_card5'] = train_trans['card5'][train_trans['card5'].notna()].apply(lambda x: len(str(int(x))))
#train_trans['length_card5'].fillna(0, inplace = True)

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
#train_trans['isFraud'] = train_trans['isFraud'].astype(int)

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