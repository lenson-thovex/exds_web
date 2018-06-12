import pandas as pd
import numpy as np
from datetime import datetime
from calendar import monthrange
from itertools import chain

# Functions
def calculate_growth(df,start,end):
	start_value = df.iloc[np.searchsorted(df.index, start),7]
	end_value = df.iloc[np.searchsorted(df.index, end, side='right')-1,7]
	return (end_value-start_value)/start_value

def calculate_monthly_growth(df,date_tuple):
	start = datetime(date_tuple[0],date_tuple[1],1)-pd.Timedelta('1D')
	end = datetime(date_tuple[0],date_tuple[1],monthrange(date_tuple[0],date_tuple[1])[1])
	if any([df.index.normalize().contains(x) for x in pd.date_range(start,end)]):
		return calculate_growth(df,max(start,df.index[0]),min(end,df.index[-1]))
	return np.NaN

def calculate_yearly_growth(df,year):
	start = datetime(year,1,1)
	end = datetime(year,12,31)
	if df.index.year.contains(year):
		return calculate_growth(df,max(start,df.index[0]),min(end,df.index[-1]))

def performance_per_tf(df,days):
	end = pd.Timestamp('today').normalize()
	start = min((end - pd.Timedelta(str(days) + 'D')),df.index[-1])
	return calculate_growth(df,start,end)

def performance_ytd(df):
	end = pd.Timestamp('today').normalize()
	start = datetime(end.year,1,1)
	return calculate_growth(df,start,end)

def drawdowns(df):
	maximums = np.maximum.accumulate(df['balance'])
	drawdowns = 1 - (df['balance'] / maximums)
	return drawdowns

def calculate_cagr(df):
	number_of_years = ((df.index[-1] - df.index[0]).days/365)
	return pow((df.iloc[-2,7]/df.iloc[0,7]),(1/number_of_years))-1

def MAR(df):
	days = (df.index[-1] - df.index[0]).days
	annual_return = performance_per_tf(df,days)/(days/360)
	max_dd = float(np.max(drawdowns(df)))
	return float(annual_return/max_dd)

def CAL_MAR(df):
	days = min(1080,(df.index[-1] - df.index[0]).days-1)
	mask = (df.index >= pd.Timestamp('today')-pd.Timedelta(str(days) + 'D'))
	periods_return = performance_per_tf(df,days)/(days/360)
	max_dd = float(np.max(drawdowns(df[mask])))
	return float(periods_return/max_dd)

def sharpe_ratio(df,days):
	timeframe_return = performance_per_tf(df,days)*100
	vola = df.balance.rolling(min(days,(df.index[-1] - df.index[0]).days-1)).std(ddof=0)[-1]
	risk_free_return = 0.0
	return (timeframe_return-risk_free_return)/vola

def sterling_ratio(df,days):
	mask = (df.index >= pd.Timestamp('today')-pd.Timedelta(str(days) + 'D'))
	cagr = calculate_cagr(df[mask])
	years = list(dict.fromkeys(df.index.year))
	sum_of_dd = 0
	for year in years:
		mask = (df.index >= datetime(year,1,1)) & (df.index <= datetime(year,12,31))
		sum_of_dd += np.max(drawdowns(df[mask]))
	avrg_dd = (sum_of_dd*-1)/len(years)
	return cagr/abs(avrg_dd-0.1)




class metatrader_df:

	#def __init__(self,db,start,end,initial_balance):
	#	self.df = #client.query(db,start,end)
	#	self.multiplyer = df.iloc[0,8]/initial_balance
	
	def __init__(self,initial_balance):
		path = 'C:\\Users\\Lenson\\notebooks\\trading\\tester_factsheets\\'
		file = 'test_daten_systeme_lennart_lange reihe.txt'
		df = pd.read_csv(path+file,delimiter='\t',header=None)
		df = df.drop(0,axis=1)
		df.columns = ['datetime','type','order#','volume','price','s/l','t/p','return','balance']
		df['datetime'] = pd.to_datetime(df['datetime'])
		self.df = df
		self.multiplyer = df.iloc[0,8]/initial_balance
		self.initial_balance = initial_balance

	def calculate_stats(self):
		#if inlfux can't store timestamps
		df = self.df

		df['datetime'] = pd.to_datetime(df['datetime'])
		
		#remove pending orders and modifies
		df = df[df['type'].apply(lambda x: x in ['buy','sell','s/l','t/p','close'])]
		df = df.reset_index(drop=True)

		df_ts_index = self.df.copy()
		df_ts_index = df_ts_index.set_index(pd.DatetimeIndex(df_ts_index['datetime']))
		df_ts_index = df_ts_index.drop('datetime',axis=1)

		df_close_only = df[df['return']!=0]
		df_close_only.reset_index(drop=True)

		# Calculations
		loss_streak = np.array([None]*len(df_close_only))
		loss_streak[-1] = 0

		win_streak = np.array([None]*len(df_close_only))
		win_streak[-1] = 0

		for i in range(len(df_close_only)-1):
			if df_close_only.iloc[i,7] < 0:
				loss_streak[i] = loss_streak[i-1]+1
				win_streak[i] = 0
			else:
				loss_streak[i] = 0
				win_streak[i] = win_streak[i-1]+1

		biggest_win_streak = max(win_streak)
		biggest_loss_streak = max(loss_streak)

		win_streak_end = np.argmax(win_streak)
		win_frame = df_close_only.iloc[win_streak_end-biggest_win_streak+1:win_streak_end+1,7]

		loss_streak_end = np.argmax(loss_streak)
		loss_frame = df_close_only.iloc[loss_streak_end-biggest_loss_streak+1:loss_streak_end+1,7]

		biggest_consecutive_return = win_frame.sum()*self.multiplyer
		biggest_consecutive_loss = loss_frame.sum()*self.multiplyer

		len_bigger_1 = np.vectorize(lambda x: len(x)>1)

		#split win_streak on zeros
		win_streaks = np.split(win_streak,np.where(win_streak==0)[0][1:])
		real_win_streaks = np.extract(len_bigger_1(win_streaks),win_streaks)

		win_streak_sum = 0
		for streak in real_win_streaks:
			win_streak_sum += streak[-1]

		average_win_streak = np.round(win_streak_sum/len(real_win_streaks),0)

		#split loss_streak on zeros
		loss_streaks = np.split(loss_streak,np.where(loss_streak==0)[0][1:])
		real_loss_streaks = np.extract(len_bigger_1(loss_streaks),loss_streaks)

		loss_streak_sum = 0
		for streak in real_loss_streaks:
			loss_streak_sum += streak[-1]

		average_loss_streak = np.round(loss_streak_sum/len(real_loss_streaks),0)

		net_profit = (df.iloc[-1,8] - df.iloc[0,8])*self.multiplyer
		gross_profit = (df[df['return']>0.0]['return'].sum())*self.multiplyer
		gross_loss = (df[df['return']<0.0]['return'].sum())*self.multiplyer
		max_dd = float(np.max(drawdowns(df)))

		trades = len(df_close_only)
		short_trades = len(df[df['type'] == 'sell'])
		long_trades = len(df[df['type'] == 'buy'])
		won_trades = len(df[df['return']>0])
		pc_won_trades = won_trades/max(1,trades)
		lost_trades = len(df[df['return']<0])
		pc_lost_trades = lost_trades/max(1,trades)
		biggest_profit = max(df['return'])*self.multiplyer
		biggest_loss = min(df['return'])*self.multiplyer
		avg_win = df[df['return']>0.0]['return'].mean()*self.multiplyer
		avg_loss = df[df['return']<0.0]['return'].mean()*self.multiplyer

		buy_orders = df[df['type'] == 'buy']['order#']
		sell_orders = df[df['type'] == 'sell']['order#']

		grouped_orders = df.groupby(['order#']).sum()

		pc_won_short_trades = len(grouped_orders[(grouped_orders.index.isin(sell_orders)) & (grouped_orders['return']>0.0)])/max(1,short_trades)
		pc_won_long_trades = len(grouped_orders[(grouped_orders.index.isin(buy_orders)) & (grouped_orders['return']>0.0)])/max(1,long_trades)

		print('initial_balance: ' + str(self.initial_balance))
		print('trades: ' + str(trades))
		print('short_trades: ' + str(short_trades))
		print('pc_won_short_trades: ' + str(pc_won_short_trades))
		print('long_trades: ' + str(long_trades))
		print('pc_won_long_trades: ' + str(pc_won_long_trades))
		print('won_trades: ' + str(won_trades))
		print('pc_won_trades: ' + str(pc_won_trades))
		print('lost_trades: ' + str(lost_trades))
		print('pc_lost_trades: ' + str(pc_lost_trades))
		print('net_profit: ' + str(net_profit))
		print('gross_profit: ' + str(gross_profit))
		print('gross_loss: ' + str(gross_loss))
		print('max_dd: ' + str(max_dd))
		print('biggest_profit: ' + str(biggest_profit))
		print('biggest_loss: ' + str(biggest_loss))
		print('avg_win: ' + str(avg_win))
		print('avg_loss: ' + str(avg_loss))
		print('biggest_win_streak: ' + str(biggest_win_streak))
		print('biggest_consecutive_return: ' + str(biggest_consecutive_return))
		print('biggest_loss_streak: ' + str(biggest_loss_streak))
		print('biggest_consecutive_loss: ' + str(biggest_consecutive_loss))
		print('average_win_streak: ' + str(average_win_streak))
		print('average_loss_streak: ' + str(average_loss_streak))
		print('\n')
		print('ytd_profit: ' + str(performance_ytd(df_ts_index)))
		print('3m_profit: ' + str(performance_per_tf(df_ts_index,90)))
		print('6m_profit: ' + str(performance_per_tf(df_ts_index,180)))
		print('12m_profit: ' + str(performance_per_tf(df_ts_index,360)))

		print('250d_vola: ' + str(df.balance.rolling(250).std(ddof=0).iloc[-1]/100))
		print('mar: ' + str(MAR(df_ts_index)))
		print('calmar: ' + str(CAL_MAR(df_ts_index)))
		print('sharpe_ratio: ' + str(sharpe_ratio(df_ts_index,3*365)))
		print('sterling_ratio: ' + str(sterling_ratio(df_ts_index,3*365)))

	def calculate_table(self):
		df_ts_index = self.df.copy()
		df_ts_index = df_ts_index.set_index(pd.DatetimeIndex(df_ts_index['datetime']))
		df_ts_index = df_ts_index.drop('datetime',axis=1)

		#create an empty df with full size (6 years x 12 months + 13th month for yearly growth)
		years = list(range(pd.Timestamp('today').year-5,pd.Timestamp('today').year+1))
		months = list(range(1,13))
		year_labels = list(chain.from_iterable(list([x]*12 for x in range(6))))
		month_labels = list(list(range(12))*6)
		labels=[year_labels] + [month_labels]
		monthly_index = pd.MultiIndex(levels=[years,months],labels=labels,names=['years','months'])
		monthly_pnl = pd.DataFrame(index=monthly_index,columns=['pnl'])

		temp_pnl = []

		for index, df_select in monthly_pnl.groupby(level=[0, 1]):
			temp_pnl.append(calculate_monthly_growth(df_ts_index,index))

		monthly_pnl['pnl'] = temp_pnl
		monthly_pnl.fillna(0,inplace=True)
		monthly_pnl = monthly_pnl.unstack(level=[0])


		for year in years:
			monthly_pnl.loc['yearly',('pnl',year)] = (calculate_yearly_growth(df_ts_index,year))

		monthly_pnl = monthly_pnl.fillna(0)
		
		return monthly_pnl



