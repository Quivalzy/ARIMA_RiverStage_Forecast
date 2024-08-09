import numpy as np
from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# @title Baca Data
indir='/Modul_4/'
datatma = pd.read_excel(indir + 'Copy of Data TMA RAW.xlsx', sheet_name = 'Nanjung', parse_dates=True)
datatma.head()

datatma['Date'] = pd.to_datetime(datatma['Date'])
datatmaTS = datatma.set_index('Date')
datatmaTS = datatmaTS.loc['2019-04-01':'2019-06-09']
datatmaTS = datatmaTS.reset_index()

tgl = datatmaTS['Date'].dt.strftime('%d-%m')
tma = datatmaTS['TMA']

fig,ax=plt.subplots()
ax.plot(tgl,tma,'-o', color = 'darkseagreen')
ax.axhline(tma.mean(), linestyle = '--', color = 'orange', label = 'Rata-Rata')
ax.set_xticks(np.arange(0, 70, 9))
ax.set_xlabel('Tanggal')
ax.set_ylabel('Tinggi Muka Air')
ax.set_title('Tinggi Muka Air di Outlet Nanjung 1 April 2019 - 9 Juni 2019')
ax.legend()
plt.savefig(indir + 'Plot_TMA.png')

tmaDiff1 = np.diff(tma)

fig, ax = plt.subplots(figsize=(10, 6))  # Atur ukuran gambar di sini
plot_acf(tmaDiff1, ax=ax, lags=14)
ax.set_xlabel('Lag')
ax.set_ylabel('Auto-korelasi')
plt.savefig(indir + 'ACF.png')

fig, ax = plt.subplots(figsize=(10, 6))  # Atur ukuran gambar di sini
plot_pacf(tmaDiff1, ax=ax, lags=14)
ax.set_xlabel('Lag')
ax.set_ylabel('Auto-korelasi')
plt.savefig(indir + 'PACF.png')

par = np.polyfit(tma[1::],tma[0:-1], 1)
fx = np.poly1d(par)
x=np.linspace(tma.min(),tma.max())
y=fx(x)

fig,ax=plt.subplots()
ax.plot(tma[1::],tma[0:-1],'.', color = 'red')
ax.plot(x,y, color = 'navy')
ax.set_xlabel('TMA (t)')
ax.set_ylabel('TMA (t+1)')
ax.set_title('Plot Autoregressive 1st Order')
plt.savefig(indir + 'AutoReg1.png')
print("f(x)=%.2fx+%.2f"%(par[0],par[1]))

phi = par[0]
eps = (tma[1::]-tma.mean())-phi*(tma[0:-1]-tma.mean())
eps_std = np.std(eps, ddof=1)
eps_2 = eps_std**2
print(eps_2, phi)

print("x(t+1)-mu=%.2f(x(t)-mu)+%.2f"%(phi,eps_2))

# Specify the order of the ARIMA model
order = (1, 0, 0)  # AR(1) model (p,d,q)

# Fit the ARIMA model
model = ARIMA(tma, order=order)
model_fit = model.fit()

# Print model parameters
print(model_fit.params)

train, test = tma[0:tma.size-14], tma[tma.size-14:]
print(train.size,test.size)

model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
model_fit.params
prediksi = model_fit.predict(start=57, end=70, dynamic=False)
print(prediksi.size)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test,prediksi))
print('RMSE periode testing: %.3f' % rmse)
print('')

# Plot Hasil Prediksi
fig,ax=plt.subplots()
ax.plot(tgl[tma.size-14:],test,'-o', color = 'forestgreen')
ax.plot(tgl[tma.size-14:],prediksi,'-o', color = 'crimson')
ax.legend(['Observasi','Prediksi'])
ax.set_xlabel('Tanggal')
ax.set_ylabel('TMA')
ax.set_title('Hasil Prediksi TMA Harian Outlet Nanjung 27 Mei 2019 - 9 Juni 2019')
ax.set_xticks(np.arange(1, 14, 2))
plt.savefig(indir + 'Hasil Predict.png')

historis=[x for x in train]
prediksi2 = list()
for t in test.index:
  model = ARIMA(historis, order=(2,0,1))
  model_fit = model.fit()
  output = model_fit.forecast()
  yhat = output[0]
  prediksi2.append(yhat)
  obs = test[t]
  historis.append(obs)
  print('prediksi=%.2f, observasi=%.2f'% (yhat, obs))
len(prediksi2)

rmse = np.sqrt(mean_squared_error(test,prediksi2))
print('RMSE periode testing: %.3f' % rmse)

# Plot
fig,ax=plt.subplots()
ax.plot(tgl[tma.size-14:],test,'-o', color = 'forestgreen')
ax.plot(tgl[tma.size-14:],prediksi2,'-o', color = 'crimson')
ax.legend(['Observasi','Prediksi'])
ax.set_xlabel('Tanggal')
ax.set_ylabel('TMA')
ax.set_title('Hasil Rolling Forecast TMA Harian Outlet Nanjung 27 Mei 2019 - 9 Juni 2019')
ax.set_xticks(np.arange(1, 14, 2))
plt.savefig(indir + 'Hasil Rolling.png')