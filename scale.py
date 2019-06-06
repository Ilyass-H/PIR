import pandas as pd

#pd.set_option('display.max_columns', 1000)
#pd.set_option('display.max_rows', 100000)
#pd.set_option('display.width', 1000)

# training set segment ids : 1 - 16275  / size = 16275
# validation set segment ids : 16276 - 21896 / size = 5621
# training test size = 21896
# test set segment ids : 21899 - 41183 / size = 10279

d_train = pd.read_csv("DH8D_train.csv.xz")
d_valid = pd.read_csv("DH8D_valid.csv.xz")
d_test = pd.read_csv("DH8D_test.csv.xz")

d_train = d_train[['segment','taskalman','baroaltitudekalman','ukalman','vkalman','heading','lat','lon','tempkalman','massFutur','distance_from_dep','trip_distance']]
d_valid = d_valid[['segment','taskalman','baroaltitudekalman','ukalman','vkalman','heading','lat','lon','tempkalman','massFutur','distance_from_dep','trip_distance']]
d_test = d_test[['segment','taskalman','baroaltitudekalman','ukalman','vkalman','heading','lat','lon','tempkalman','massFutur','distance_from_dep','trip_distance']]

d_train = d_train.dropna()
d_valid = d_valid.dropna()
d_test = d_test.dropna()

d_train_test = d_train.append(d_valid)

segmentCol_train = d_train['segment']
segmentCol_valid = d_valid['segment']
segmentCol_test = d_test['segment']
segmentCol_train_test = d_train_test['segment']

d_train = d_train.drop(['segment'],axis=1)
d_valid = d_valid.drop(['segment'],axis=1)
d_test = d_test.drop(['segment'],axis=1)
d_train_test = d_train_test.drop(['segment'],axis=1)


d_train = (d_train.subtract(d_train.mean())).divide(d_train.std())
d_valid = (d_valid.subtract(d_valid.mean())).divide(d_valid.std())
d_test = (d_test.subtract(d_test.mean())).divide(d_test.std())
d_train_test = (d_train_test.subtract(d_train_test.mean())).divide(d_train_test.std())

d_train = pd.concat([segmentCol_train,d_train],axis=1)
d_valid  = pd.concat([segmentCol_valid,d_valid],axis=1)
d_test  = pd.concat([segmentCol_test,d_test],axis=1)
d_train_test = pd.concat([segmentCol_train_test,d_train_test],axis=1)


f_train = open("scaledTrainData.csv","w")
f_valid = open("scaledValidData.csv","w")
f_test = open("scaledTestData.csv","w")
f_train_test = open("scaledTrainTestData.csv","w")

f_train.write(d_train.to_csv(index=False))
f_valid.write(d_valid.to_csv(index=False))
f_test.write(d_test.to_csv(index=False))
f_train_test.write(d_train_test.to_csv(index=False))

f_train.close()
f_valid.close()
f_test.close()
f_train_test.close()
