import zipline as zp
import quandl as qd
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pytz
import os.path
import string
import time
import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators







# Grab the data for each company called
def init_data(comp, interval):
		
		# Input all the technical indicators and the stock times series data

		# Initialize the AlphaVantage API call functions
		ts = TimeSeries(key='L2O2TYTG382ETN0N', output_format='pandas')
		ti = TechIndicators(key='L2O2TYTG382ETN0N', output_format='pandas')


		# Other keys: QB5APLD84E50TQAC, L2O2TYTG382ETN0N

		# Fetch the simple moving average (SMA) values
		main_data, meta_data = ti.get_sma(symbol=comp,interval='60min', time_period=20, series_type='close')
		
		time.sleep(3)

		# Fetch the exponential moving average (EMA) values
		ema_data, meta_data = ti.get_ema(symbol=comp,interval='60min', time_period=20, series_type='close')
		main_data['EMA'] = ema_data
		
		time.sleep(3)
		"""
		# Fetch the weighted moving average (WMA) values
		wma_data, meta_data = ti.get_wma(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['WMA'] = wma_data
		
		# Fetch the double exponential moving agerage (DEMA) values
		dema_data, meta_data = ti.get_dema(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['DEMA'] = dema_data
		
		# Fetch the triple exponential moving average (TEMA) values
		tema_data, meta_data = ti.get_tema(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['TEMA'] = tema_data
		
		# Fetch the triangular moving average (TRIMA) values 
		trima_data, meta_data = ti.get_trima(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['TRIMA'] = trima_data

		# Fetch the Kaufman adaptive moving average (KAMA) values
		kama_data, meta_data = ti.get_kama(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['KAMA'] = kama_data		

		# Fetch the MESA adaptive moving average (MAMA) values
		mama_data, meta_data = ti.get_mama(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['MAMA'] = mama_data['MAMA']
		main_data['FAMA'] = mama_data['FAMA']

		# Fetch the triple exponential moving average (T3) values
		t3_data, meta_data = ti.get_t3(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['T3'] = t3_data	
		"""


		# Fetch the moving average convergence / divergence (MACD) values
		macd_data, meta_data = ti.get_macd(symbol=comp,interval='60min', series_type='close')
		main_data['MACD'] = macd_data['MACD']
		main_data['MACD_Hist'] = macd_data['MACD_Hist']
		main_data['MACD_Signal'] = macd_data['MACD_Signal']

		time.sleep(3)
		"""		
		# Fetch the moving average convergence / divergence values with controllable moving average type
		macdext_data, meta_data = ti.get_macdext(symbol=comp,interval='60min', series_type='close')
		main_data['MACDEXT'] = macdext_data['MACD']
		main_data['MACDEXT_Hist'] = macdext_data['MACD_Hist']
		main_data['MACDEXT_Signal'] = macdext_data['MACD_Signal']
		"""

		# Fetch the stochastic oscillator (STOCH) values
		stoch_data, meta_data = ti.get_stoch(symbol=comp,interval='60min')
		main_data['SlowK'] = stoch_data['SlowK']
		main_data['SlowD'] = stoch_data['SlowD']

		time.sleep(3)
		"""
		# Fetch the stochastic fast (STOCHF) values
		stochf_data, meta_data = ti.get_stochf(symbol=comp,interval='60min')
		main_data['FastK'] = stochf_data['FastK']
		main_data['FastD'] = stochf_data['FastD']
		"""


		# Fetch the relative strength index (RSI) values
		rsi_data, meta_data = ti.get_rsi(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['RSI'] = rsi_data

		time.sleep(3)
		"""
		# Fetch the stochastic relative strength index (STOCHRSI) values
		stochrsi_data, meta_data = ti.get_stochrsi(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['STOCHRSI_FastK'] = stochrsi_data['FastK']
		main_data['STOCHRSI_FastD'] = stochrsi_data['FastD']

		# Fetch the Williams' %R (WILLR) values
		willr_data, meta_data = ti.get_willr(symbol=comp,interval='60min', time_period=10)
		main_data['WILLR'] = willr_data
		"""



		# Fetch the average directional movement index (ADX) values
		adx_data, meta_data = ti.get_adx(symbol=comp,interval='60min', time_period=20)
		main_data['ADX'] = adx_data

		time.sleep(3)
		"""
		# Fetch the average directional movement index rating (ADXR) values
		adxr_data, meta_data = ti.get_adxr(symbol=comp,interval='60min', time_period=10)
		main_data['ADXR'] = adxr_data

		# Fetch the absolute price oscillator (APO) values
		apo_data, meta_data = ti.get_apo(symbol=comp,interval='60min', series_type='close')
		main_data['APO'] = apo_data

		# Fetch the percentage price oscillator (PPO) values
		ppo_data, meta_data = ti.get_ppo(symbol=comp,interval='60min', series_type='close')
		main_data['PPO'] = ppo_data

		# Fetch the momentum (MOM) values
		mom_data, meta_data = ti.get_mom(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['MOM'] = mom_data

		# Fetch the balance of power (BOP) values
		bop_data, meta_data = ti.get_bop(symbol=comp,interval='60min')
		main_data['BOP'] = bop_data
		"""

		# Fetch the commodity channel index (CCI) values
		cci_data, meta_data = ti.get_cci(symbol=comp,interval='60min', time_period=20)
		main_data['CCI'] = cci_data

		time.sleep(3)

		"""
		# Fetch the Chande momentum oscillator (CMO) values
		cmo_data, meta_data = ti.get_cmo(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['CMO'] = cmo_data

		# Fetch the rate of change (ROC) values
		roc_data, meta_data = ti.get_roc(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['ROC'] = roc_data


		# Fetch the rate of change ratio (ROCR) values
		rocr_data, meta_data = ti.get_rocr(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['ROCR'] = rocr_data

		time.sleep(5)
		"""
		# Fetch the Aroon (AROON) values
		aroon_data, meta_data = ti.get_aroon(symbol=comp,interval='60min', time_period=20)
		main_data['Aroon Down'] = aroon_data['Aroon Down']
		main_data['Aroon Up'] = aroon_data['Aroon Up']

		time.sleep(3)

		"""
		# Fetch the Aroon oscillator (AROONOSC) values
		aroonosc_data, meta_data = ti.get_aroonosc(symbol=comp,interval='60min', time_period=10)
		main_data['AROONOSC'] = aroonosc_data

		# Fetch the money flow index (MFI) values
		mfi_data, meta_data = ti.get_mfi(symbol=comp,interval='60min', time_period=10)
		main_data['MFI'] = mfi_data

		# Fetch the 1-day rate of change of a triple smooth exponential moving average (TRIX) values
		triX_train_data['AAPL'], meta_data = ti.get_trix(symbol=comp,interval='60min', time_period=10, series_type='close')
		main_data['TRIX'] = triX_train_data['AAPL']

		# Fetch the ultimate oscillator (ULTOSC) values
		ultosc_data, meta_data = ti.get_ultsoc(symbol=comp,interval='60min', time_period=10)
		main_data['ULTOSC'] = ultosc_data

		# Fetch the directional movement index (DX) values
		dX_train_data['AAPL'], meta_data = ti.get_dx(symbol=comp,interval='60min', time_period=10)
		main_data['DX'] = dX_train_data['AAPL']
		"""


		"""
		# Fetch the Chaikin A/D line (AD) value
		ad_data, meta_data = ti.get_ad(symbol=comp,interval='60min')
		main_data['AD'] = ad_data
		"""

		# Fetch the on balance volume (OBV) values
		obv_data, meta_data = ti.get_obv(symbol=comp,interval='60min')
		main_data['OBV'] = obv_data


		intraday_data, meta_data = ts.get_intraday(symbol=comp,interval='60min', outputsize='full')
		

		intraday_data.index = pd.Index([index[:-3] for index in intraday_data.index], name='date')
		#intraday_data.set_index('date')



		main_data = pd.concat([main_data, intraday_data], axis=1)
		print(main_data.index)

		print(main_data.head())

		main_data = main_data.dropna()
		main_data.index.name = 'date'

		company = comp
		main_data.to_csv(f'./rsc/{company}_AV_data.csv', sep = ',')

		time.sleep(1)

		print(comp)

		time.sleep(5)





if __name__ == '__main__':
	companies = ['AAPL', 'MSFT', 'AMZN', 'INTC', 'TSLA', 'GOOG']



	for comp in companies:
		init_data(comp, '60min')