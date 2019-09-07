# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:21:02 2019

@author: pozir
"""

import pandas as pd
import pickle

train_ident = pd.read_csv('data/train_identity.csv')
train_trans = pd.read_csv('data/train_transaction.csv')

train_ident = pd.merge(train_trans, train_ident, how = 'left', on ='TransactionID')
del train_trans
#==============================================================================
# Numeric data
#==============================================================================

numeric_var = ['id_01','id_02','id_03','id_04', 'id_05','id_06','id_07','id_08',
 'id_09','id_10','id_11','id_13', 'id_14','id_17','id_18','id_19',
 'id_20','id_21','id_22','id_24','id_25','id_26']

num_data = train_ident[numeric_var].describe().T[1:]

nas = []
unique_v = []

for i in num_data.index:
    nas.append(train_ident[i].isna().sum())
    unique_v.append(len(train_ident[i].value_counts()))
    
num_data['Nas'] = nas
num_data['unique_v'] = unique_v

numeric_var = list(num_data[num_data['unique_v'] >10].index)

'''
                   Masking missing values in numerical data
                             min - (max - min)**2
'''
num_data['Nas_mask'] = num_data['min'] - (num_data['max'] - num_data['min'])**2

train_ident['id_32'] = train_ident['id_32'][train_ident['id_32'].notna()].apply(lambda x:
    str(x))
train_ident['id_32'].fillna('notknown', inplace = True)
#==============================================================================
# Categorical data
#==============================================================================

categ_var = ['id_12', 'id_15','id_16','id_23','id_27', 'id_28','id_29', 'id_30',
 'id_31','id_32','id_33', 'id_34','id_35','id_36', 'id_37','id_38',
 'DeviceType','DeviceInfo']
#for i in train_ident.columns:
#    if i not in numeric_var and i != 'TransactionID':
#        categ_var.append(i)
#        
train_ident[categ_var].info()

unique_values = []

for i in  train_ident[categ_var]:
    unique_values.append(len(train_ident[i].value_counts()))
    
nas = []

for i in train_ident[categ_var]:
    nas.append((train_ident[i].isna().sum()))

cat_data = pd.DataFrame({'unique_v' : unique_values, 'Nas': nas})
cat_data.index = categ_var

modify_cat = list(cat_data[cat_data['unique_v'] > 10 ].index)

#==============================================================================
# Replacing Nas with masked values (numerical data)
#==============================================================================

for i in train_ident[numeric_var]:
    train_ident[i].fillna(num_data['Nas_mask'].loc[i], inplace = True)

#==============================================================================
# Modifying 'id_30', 'id_31', 'id_33', 'DeviceInfo'   
#==============================================================================
'''
                                OS types id_30
'''
train_ident['id_30'] = train_ident['id_30'][train_ident['id_30'].notna()].apply(
        lambda x: x.split(' ')[0])

train_ident['id_30'].fillna('notknown', inplace = True)

'''
                                Browser type id_31
'''

train_ident['mobile_devise'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if ((x.find('mobile') > -1) | (x.lower().find('samsung') > -1)
                        | (x.lower().find('nokia') > -1)) else 0)     
     
train_ident['mobile_devise'].fillna(-1, inplace = True)
numeric_var.append('mobile_devise')

train_ident['tablet'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('tablet') > -1) else 0)     

train_ident['tablet'].fillna(-1, inplace = True)  
numeric_var.append('tablet')   

train_ident['chrome'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('chrome') > -1) else 0)     

train_ident['chrome'].fillna(-1, inplace = True)   
numeric_var.append('chrome')

train_ident['safari'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('safari') > -1) else 0)     

train_ident['safari'].fillna(-1, inplace = True)
numeric_var.append('safari')   

train_ident['generic'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.lower().find('generic') > -1) else 0)     

train_ident['generic'].fillna(-1, inplace = True) 
numeric_var.append('generic')  

train_ident['desktop'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('desktop') > -1) else 0)     

train_ident['desktop'].fillna(-1, inplace = True)  
numeric_var.append('desktop') 

train_ident['android'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('android') > -1) else 0)     

train_ident['android'].fillna(-1, inplace = True)
numeric_var.append('android')   

train_ident['edge'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('edge') > -1) else 0)     

train_ident['edge'].fillna(-1, inplace = True)
numeric_var.append('edge')   

train_ident['firefox'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('firefox') > -1) else 0)     

train_ident['firefox'].fillna(-1, inplace = True)
numeric_var.append('firefox')   

train_ident['ie'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('ie') > -1) else 0)     

train_ident['ie'].fillna(-1, inplace = True) 
numeric_var.append('ie')  


train_ident['ios'] = train_ident['id_31'][train_ident['id_31'].notna()].apply(
        lambda x: 1 if (x.find('ios') > -1) else 0)     

train_ident['ios'].fillna(-1, inplace = True) 
numeric_var.append('ios')  

categ_var.remove('id_31')

'''
                  id_33 possible area of the device
'''

train_ident['id_33_'] = train_ident['id_33'][train_ident['id_33'].notna()].apply(
        lambda x: round((int(x.split('x')[0]) * int(x.split('x')[1]) )/254))

train_ident['id_33_'].fillna(0.0, inplace = True)

numeric_var.append('id_33_')
categ_var.remove('id_33')

'''
                     DeviceInfo 
'''
categ_var.remove('DeviceInfo')

#==============================================================================
# Replacing Nas with masked values (categorical data data)
#==============================================================================

train_ident[categ_var] = train_ident[categ_var].fillna('notknown') 

'''
                  Final info dataset
'''
total_id = categ_var + numeric_var + ['TransactionID']
train_ident = train_ident[total_id]
train_ident.to_pickle('data/id.pickle')
