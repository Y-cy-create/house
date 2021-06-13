import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
#讀取原始train/test資料
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

#將兩資料集合併
frames = [df1, df2]
df = pd.concat(frames)
print(df)
#砍掉NA太多之欄位
df = df.drop(['Id'], axis=1)
df = df.drop(['Alley'], axis=1)
df = df.drop(['PoolQC'], axis=1)
df = df.drop(['Fence'], axis=1)
df = df.drop(['MiscFeature'], axis=1)

#加上「屋齡」之欄位
df['house_year'] = df['YrSold'] - df['YearBuilt']
# print(df['house_year'])


#將剩下有少數NA之欄位組成一個list，方便處理
null = df.columns[df.isna().any()].tolist()

#但是SalePrice本就有一半(test)是缺失值不能包含在此
null.remove('SalePrice')


# print(df.dtypes)
# key = df.columns.tolist()
# value = df.dtypes.tolist()
# print(value)


#製造2個空list，以便分類數值資料以及類別資料
catgory = []

number = []


#dtypes若為float64 或 int64，就是數值欄位，就併入number，否則併入catgory
for y in df.columns:
    if df[y].dtype == np.float64 or df[y].dtype == np.int64:
        number.append(y)

    else:
        catgory.append(y)
# data = dict(zip(key, value))
# print(null)


#進一步找出數值和類別兩種資料類型中，有NA之欄位
cat_null = [i for i in catgory if i in null]
number_null = [i for i in number if i in null]

# null = ['LotFrontage', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']

# scaler = MinMaxScaler()



#類別資料之NA，以眾數取代
for i in cat_null:
    df[i] = df[i].fillna(df[i].mode())


#數值資料之NA，以中位數取代
for i in number_null:
    df[i] = df[i].fillna(df[i].median())
# for i in number:

    # scaler.fit(df[i])  # only fit to training data to aviod data leakage
    # df[i] = scaler.transform(df[i])




#類別資料資料進行dummy處理
for i in catgory:
    dummy = pd.get_dummies(df[i])   #產生dummy的dataframe
    df = pd.concat([df, dummy], axis=1)  #將產生dummy的dataframe合併到原本的dataframe
    df = df.drop([i], axis=1)    #砍掉原本類別欄位
# print(cat_null)
# print(number_null)
# print(df)


#X是特徵(但是訓練時只有train的1460筆資料可以參與)
X = df[:1460].drop(['SalePrice'], axis=1)
#test的特徵預備著稍後預測(訓練完模型後)
X_test = df[1460:].drop(['SalePrice'], axis=1)


#Y是目標(但是訓練時只有train的1460筆資料可以參與)
df_temp = df[:1460]
Y = df_temp['SalePrice']

#使用Ridge演算法擬和
ridge1 = Ridge(alpha=1)
ridge1.fit(X, Y)



#隨機抽取資料預估
print(Y[0])
print(ridge1.predict(X)[0])


#重新讀取原始train/test資料，以便合併預測結果
df_train_final = pd.read_csv('train.csv')
df_test_final = pd.read_csv('test.csv')
#產生train預測結果
df_train_final['sales_prediction'] = ridge1.predict(X)
#產生test預測結果
df_test_final['sales_prediction'] = ridge1.predict(X_test)
#寫入CSV
df_train_final.to_csv('train_with_prediction.csv', index=False, encoding='big5')
df_test_final.to_csv('test_with_prediction.csv', index=False, encoding='big5')








