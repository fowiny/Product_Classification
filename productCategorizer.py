import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pyodbc
#from sklearn.pipeline import make_pipeline
#import pysftp


# Create the connection to the sql server w.r.t to the DNS set
conn = pyodbc.connect("DSN=sq02", autocommit=True)

# SQL query across the tables
sql = """

-- Includes the universal codes cat.[Product Main Group], cat.[Activity Code] AND cat.[Product Sub-Group]
select *
from (
select distinct I.No_ AS itemNo, UPPER(I.codMarchio) AS cM, I.[codReparto] AS cR, I.[codSesso] AS cS, cat.[Sport] AS sp, I.[codTipo] AS cT, cat.[Vendor No_] AS vNo, UPPER(cat.[Vendor Item No_]) AS vINo, UPPER(cat.[Item Description]) AS iD, cat.[Size Code] AS sC, cat.[Activity Code] AS aC, cat.[Product Main Group] AS pMG, cat.[Product Sub-Group] AS pSG, cat.Category
		--, I.[codSport] AS cG, cat.[Color Code] AS cC
   FROM [3A_DWH].[dbo].[DimItem] I
   inner join [3A_STAGE].[nav].[Listini - Details] cat on
       UPPER(I.[Vendor Item No_]) = UPPER(cat.[Vendor Item No_]) and UPPER(I.codMarchio) = UPPER(cat.Brand) and UPPER(I.[codSesso]) = UPPER(cat.[Sex]) and UPPER(I.[Description]) = UPPER(cat.[Item Description])
       --where cat.Category != ''
	WHERE cat.[Product Main Group] <> '' AND cat.[Activity Code] <> '' AND cat.[Product Sub-Group] <> ''
   group by I.No_, I.[codMarchio], I.[codReparto], I.[codSesso], I.[codSport], I.[codTipo], cat.[Vendor No_], cat.[Vendor Item No_], cat.[Item Description], cat.[Sex], cat.[Sport], cat.[Size Code], cat.[Activity Code], cat.[Product Main Group], cat.[Product Sub-Group], cat.Category
   --, I.[codSport], cat.[Color Code]
   )t
   group by itemNo, cM, cR, cS, sp, cT, vNo, vINo, iD, sC, ac, pMG, pSG, category
   --, cG, cC
  
"""

# Read sql query output to panda dataframe
df = pd.io.sql.read_sql(sql, conn)
# Save the DF to dataset-folder
df.to_csv('dataset/productCategorizer.csv')
####  View first values of the file
df.head()

### NULL VALUES
## replace all blank/empty cells with NaNs
df.replace('', np.nan, inplace=True)
# column-wise distribution of null values
print(df.isnull().sum())
###### Remove rows with NA values w.r.t the given columns
df.dropna(subset = [ x for x in df.columns if x != 'Category'], inplace = True)
print(len(df), '\n\n', df.isnull().sum())

#####   FEATURE ENGINEERING
### split the iD column into strings by spaces
df_iD = df.copy()
series_iD = df['iD'].str.strip().str.split(' ')
######Derive new array for first and last words on the lists........values changes from series to array to stop the NaN outputs
df_iD['iD_0'] = series_iD.apply(lambda w: w[0]).values
df_iD['iD_L'] = series_iD.apply(lambda w: w[-1]).values
##### View last values of the file => check that NaN values are not registered
df_iD.tail()

####  Delete category classes that are very few
## Create a dictionary of Category : value_count
categ_count = dict(zip(df_iD['Category'].value_counts().index, df_iD['Category'].value_counts().values))
### Create list of "Category" whose value_counts is less than 20 or as chosen
elimnate_categs = [x for x in categ_count.keys() if categ_count[x] < 20]
# Deleting rows with Category in elimnate_categs
cleanedData = df_iD[~df_iD.Category.isin(elimnate_categs)]
print ('Perecntage remaining after deletion: ', len(cleanedData)/len(df_iD))

#####   TRANSFORM CATEGORICAL FEATURES to NUMERICAL
# w is the dataframe with categorical values
def categ_maps(w):
    mapping_dict = {}
    for x, y in enumerate(w.columns):
        d = {}
        for a,b in enumerate(w.iloc[:,x].unique()):
            d.update({b: a})
        mapping_dict.update({y: d})
    return mapping_dict
# The function to map dataframe to numerical values
def df_maps(z):
    g = categ_maps(z)
    df_mapped = pd.DataFrame()
    for r in z.columns:
        df_mapped[r] = z[r].map(g[r])
    return df_mapped

# Using df_maps created function, transform the data
df_num = df_maps(cleanedData)
print (len(cleanedData), len(df_num), '\n\n', df_num.head(3))
### Test the functions
dic_obj = categ_maps(cleanedData)
print (dic_obj.keys(), '\n')
print (dic_obj['Category'], '\n\n')
### Number of data with Null values and those with values
print ('Null w.r.t numerical data : ', len(df_num[df_num['Category']==dic_obj['Category'][np.nan]]), '\n', 'Null w.r.t original data : ', len(cleanedData[cleanedData['Category'].isnull()]))

### Divied  transformed data into labelled and unlabelled
scaledLabelledData = df_num[df_num['Category'] != 0]
scaledUnlabelledData = df_num[df_num['Category'] == 0]
print (len(scaledLabelledData), len(scaledUnlabelledData))

### PERFORM CLASSIFICATION TASK
print ('PERFORMING CLASSIFICATION TASK')
## Split dataset
x_train, x_test, y_train, y_test = train_test_split(scaledLabelledData.drop(['Category'], axis=1), scaledLabelledData.loc[:,'Category'], test_size=0.25, random_state=0)

## DECISION TREE CLASSIFIER
prod = tree.DecisionTreeClassifier()
prod = prod.fit(x_train, y_train)
# Use fit_predict to fit model and obtain cluster labels: labels
pred_labels = prod.predict(x_test)
# Create a DataFrame with clusters and varieties as columns: df
dfpy = pd.DataFrame({'pred_labels': pred_labels, 'y_labels': y_test})
# Create crosstab: ct
ct = pd.crosstab(dfpy['pred_labels'], dfpy['y_labels'])
# Display ct
print(ct)

## Retrain and perform unlabelled predictions
prod_obj = tree.DecisionTreeClassifier()
prod_model = prod_obj.fit(scaledLabelledData.drop(['Category'], axis=1), scaledLabelledData.loc[:,'Category'])
### Perform prediction
predicted_unlabeled = prod_model.predict(scaledUnlabelledData.drop(['Category'], axis=1))

# Invert the dictionary
dic_keys = dic_obj['Category'].keys()
dic_values = dic_obj['Category'].values()
dic_inv = dict(zip(dic_values, dic_keys))
### Replace predicted terms with their rvivalent categories
predictions = [dic_inv[key] for key in predicted_unlabeled]
print (len(predictions), len(predicted_unlabeled), len(cleanedData[cleanedData['Category'].isnull()]))

### Put the predicted terms back to the original dataframe
df_predictions = cleanedData[cleanedData['Category'].isnull()].drop('Category', axis = 1)
df_predictions['Categ_ID'] = predicted_unlabeled
df_predictions['Category'] = predictions
df_predictions.tail(3)


### ADJUST OUTPUT ACCORDING TO REVIEWS BY DOMAIN USERS
### REVIEW by LORIS
# LORIS (cM=J11) advises that every column be labelled "YOUNG ATHLETES" ... Thereby replacing "OTHER" and "NIKE SPORTSWEAR"
df_Loris = df_predictions.copy()
cM_boolLo = (df_Loris['cM']=='J11') | (df_Loris['cM']=='N29') # Loris boolean list w.r.t J11 and N29
categ_boolLo = (df_Loris['Category']!='YOUNG ATHLETES') # Loris boolean list w.r.t NOT 'YOUNG ATHLETES'
no_categsLo = [x for x in dic_keys if x != 'YOUNG ATHLETES'] # List of all other categories but not 'YOUNG ATHLETES'
Loris = df_Loris[cM_boolLo & categ_boolLo]['Category'].replace(no_categsLo, "YOUNG ATHLETES", inplace=False)
df_Loris.update(Loris)
print (df_Loris[cM_boolLo & categ_boolLo].loc[:, ['cM', 'Categ_ID', 'Category']].drop_duplicates())

### REVIEW by EZIO
# Ezio (cM=A03, R04) advises that every column "TENNIS" and "NIKE SPORTSWEAR" be replaced with "CORE"
df_Ezio = df_Loris.copy()
cM_bool = (df_Ezio['cM']=='A03') | (df_Ezio['cM']=='R04')
categ_bool = (df_Ezio['Category']=='TENNIS') | (df_Ezio['Category']=='NIKE SPORTSWEAR')
Ezio = df_Ezio[cM_bool & categ_bool]['Category'].replace(["TENNIS", "NIKE SPORTSWEAR"], "CORE", inplace=False)
df_Ezio.update(Ezio)
print (df_Ezio[cM_bool & categ_bool].loc[:, ['cM', 'Categ_ID', 'Category']].drop_duplicates())

### Obtain dataframe for Category dictionary
# dataframe of category dictionary
df_dic = pd.DataFrame({'categ_ID':list(dic_values), 'category':list(dic_keys)}).iloc[1:, ]
df_dict = df_dic.append({'categ_ID': [dic_obj['Category']['TENNIS'], dic_obj['Category']['NIKE SPORTSWEAR']],\
                         'category': 'CORE'}, ignore_index=True)
print (df_dict)

### EXPORT OUTPUT AS EXCEL
# dataframe of category dictionary
print ("Outputing result(s)")
df_dic = pd.DataFrame({'categ_ID':list(dic_values), 'category':list(dic_keys)}).iloc[1:, ]
writer = pd.ExcelWriter('output/productCategorized.xlsx')
df_Ezio.to_excel(writer,'productCategory')
ct.to_excel(writer,'accuracyTest')
df_dict.to_excel(writer,'dictionaryCategory')
writer.save()

print ("Experiment Successfully Run", "\n", 'END')
