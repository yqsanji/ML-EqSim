# Kaggle 
# 'all.csv' choose random(mu_s, mu_d, dc) each has 5 random seed
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df_data=pd.read_csv('../input/0dlsw/all.csv',delimiter=' ',nrows=25000)
t_train=np.array(df_data['t'].values.tolist()).reshape(1,-1)
mus_train=np.array(df_data['mu_s(imus)'].values.tolist()).reshape(1,-1)
X1_train=np.append(t_train,mus_train,axis=0)
mud_train=np.array(df_data['mu_d(imud)'].values.tolist()).reshape(1,-1)
X2_train=np.append(X1_train,mud_train,axis=0)
dc_train=np.array(df_data['dc(idc)'].values.tolist()).reshape(1,-1)
X=np.transpose(np.append(X2_train,dc_train,axis=0))
print(X.shape)

y=np.array(df_data['V'].values.tolist()).ravel()
print(y.shape)

rf=RandomForestRegressor()
rf.fit(X,y)

df_test_data=pd.read_csv('../input/0dlsw/all.csv',delimiter=' ',skiprows=50000,nrows=5000)

t_test=np.array(df_test_data.iloc[:,6].values.tolist()).reshape(1,-1)
mus_test=np.append(t_test,np.array(df_test_data.iloc[:,0].values.tolist()).reshape(1,-1),axis=0)
mud_test=np.append(mus_test,np.array(df_test_data.iloc[:,1].values.tolist()).reshape(1,-1),axis=0)
dc_test=np.append(mud_test,np.array(df_test_data.iloc[:,2].values.tolist()).reshape(1,-1),axis=0)
testdata=np.transpose(dc_test)
print(testdata.shape)

y_test=rf.predict(testdata)
print(rf.score(testdata,y_test))

plt.plot(X[:5000,0],y_test)
