import pandas as pd

#pd.set_option('display.max_columns', 1000)
#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.width', 1000)

# training set segment ids : 1 - 11026
# validation set segment ids : 16276 - 21896

d_train = pd.read_csv("DH8D_train.csv.xz")
d_valid = pd.read_csv("DH8D_valid.csv.xz")

d_train = d_train[['segment','taskalman','baroaltitudekalman','ukalman','vkalman','heading','lat','lon','tempkalman','massFutur']]
d_valid = d_valid[['segment','taskalman','baroaltitudekalman','ukalman','vkalman','heading','lat','lon','tempkalman','massFutur']]

d_train = d_train.dropna()
d_valid = d_valid.dropna()

segmentCol_train = d_train['segment']
segmentCol_valid = d_valid['segment']

d_train = d_train.drop(['segment'],axis=1)
d_valid = d_valid.drop(['segment'],axis=1)

d_train = (d_train.subtract(d_train.mean())).divide(d_train.std())
d_valid = (d_valid.subtract(d_valid.mean())).divide(d_valid.std())

d_train = pd.concat([segmentCol_train,d_train],axis=1)
d_valid = pd.concat([segmentCol_valid,d_valid],axis=1)

f_train = open("scaledTrainData.csv","w")
f_valid = open("scaledCVData.csv","w")
f_train.write(d_train.to_csv(index=False))
f_valid.write(d_valid.to_csv(index=False))
f_train.close()
f_valid.close()