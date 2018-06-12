import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from calendar import monthrange
import openpyxl
from shutil import copy
import itertools
from os import listdir, remove
from os.path import isfile, join

### get the name from wikifolio export file
#working_folder = 'D:\\wikifolio_factsheets\\'
working_folder = 'C:\\Users\\Lenson\\notebooks\\trading\\wikifolio_factsheets\\'
onlyfiles = [f for f in listdir(working_folder) if isfile(join(working_folder, f))]
raw_csv = onlyfiles[0]

###trimming and arranging of strings and write back to csv

with open(working_folder + raw_csv) as f:
    s = f.read() + '\n'

s = s.replace('\x00', '')
s = s.replace('\"','')
s = s.replace(',','.')
s = s.split('\n')
system_name = s[1].replace(' ','_')

s = s[5:]
s = '\n'.join(s)
file_name = system_name + '_' + str(datetime.now().date()) + '.csv'

f = open(working_folder + file_name,'w')
f.write(s)

f.close()

###load and prepare data in dataframe
df = pd.read_csv(working_folder + file_name,sep=';')
df = df.drop('Time interval (min)',axis=1)
df['Begin date'] = df['Begin date'].apply(lambda x: x[:10])
df['Begin date'] = df['Begin date'].apply(lambda x: datetime.strptime(x,'%d.%m.%Y'))
df.index = pd.DatetimeIndex(df['Begin date'])

df_to_plot = df.copy()

df = df.reindex(pd.DatetimeIndex(freq='1D',start=df.index[0],end=datetime.now()))
df = df.drop('Begin date',axis=1)
df.fillna(method='ffill',inplace=True)
df['PnL'] = (df['Close']-df['Close'].shift()).fillna(df['Close'][0]-df['Open'][0])


#define functions

def calculate_yearly_growth(df,year):
    start = datetime(year,1,1)
    end = datetime(year,12,31)
    if df.index.year.contains(year):
        return calculate_growth(df,max(start,df.index[0]),min(end,df.index[-1]))

def calculate_monthly_growth(df,date_tuple):
    start = datetime(date_tuple[0],date_tuple[1],1)-pd.Timedelta('1D')
    end = datetime(date_tuple[0],date_tuple[1],monthrange(date_tuple[0],date_tuple[1])[1])
    last_day_of_month = datetime(date_tuple[0],date_tuple[1],monthrange(date_tuple[0],date_tuple[1])[1])
    if df.index.contains(start) | df.index.contains(end):
        return calculate_growth(df,max(start,df.index[0]),min(end,df.index[-1]))
    return np.NaN

def calculate_growth(df,start,end):
    start_value = df.at[start,'Close']
    end_value = df.at[end,'Close']
    return (end_value-start_value)/start_value

def calculate_cagr(df):
    number_of_years = (len(df)/365)
    return pow((df.iloc[-2,0]/df.iloc[0,0]),(1/number_of_years))-1
    
def drawdowns(df):
    maximums = np.maximum.accumulate(df.High)
    drawdowns = 1 - (df.Low / maximums)
    return drawdowns

def performance_per_tf(df,days):
    end = pd.Timestamp('today').normalize()
    start = max((end - pd.Timedelta(str(days) + 'D')),df.index[0])
    return calculate_growth(df,start,end)

def performance_ytd(df):
    end = pd.Timestamp('today').normalize()
    start = datetime(end.year,1,1)
    return calculate_growth(df,start,end)

def MAR(df):
    days = (df.index[-1] - df.index[0]).days
    annual_return = performance_per_tf(df,days)/(days/360)
    max_dd = float(np.max(drawdowns(df)))
    return float(annual_return/max_dd)

def CAL_MAR(df):
    days = min(1080,len(df)-1)
    mask = (df.index >= pd.Timestamp('today')-pd.Timedelta(str(days) + 'D'))
    periods_return = performance_per_tf(df,days)/(days/360)
    max_dd = float(np.max(drawdowns(df[mask])))
    return float(periods_return/max_dd)

def sharpe_ratio(df,days):
    timeframe_return = performance_per_tf(df,days)*100
    vola = df.Close.rolling(min(days,len(df)-1)).std(ddof=0)[-1]
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


#create an empty df with full size (6 years x 12 months)
years = list(range(pd.Timestamp('today').year-5,pd.Timestamp('today').year+1))
months = list(range(1,13))
year_labels = list(itertools.chain.from_iterable(list([x]*12 for x in range(6))))
month_labels = list(list(range(12))*6)
labels=[year_labels] + [month_labels]
monthly_index = pd.MultiIndex(levels=[years,months],labels=labels,names=['years','months'])
monthly_pnl = pd.DataFrame(index=monthly_index,columns=['pnl'])

temp_pnl = []

for index, df_select in monthly_pnl.groupby(level=[0, 1]):
    temp_pnl.append(calculate_monthly_growth(df,index))

monthly_pnl['pnl'] = temp_pnl
monthly_pnl.fillna(0,inplace=True)
monthly_pnl = monthly_pnl.unstack(level=[0])


#calculate yearly growth
yearly_pnl = []
for year in years:
    yearly_pnl.append(calculate_yearly_growth(df,year))
    
yearly_pnl = pd.DataFrame(yearly_pnl).fillna(0).transpose()
yearly_pnl.columns = years


#fill facts
monthly_sum = pd.DataFrame(df.groupby(pd.TimeGrouper('M')).sum())

results = {'net_profit' : float(df['Close'][-1]-df['Open'][0])/100,
'average_monthly_win' : monthly_pnl[monthly_pnl['pnl']>0].unstack().mean(),
'average_monthly_loss' : monthly_pnl[monthly_pnl['pnl']<0].unstack().mean(),
'months_in_calculation' : int(len(monthly_sum)),
'months_won' : int(len(monthly_sum[monthly_sum['PnL']>0]))}

results['months_won_ration'] = float(results['months_won']/results['months_in_calculation'])
results['average_monthly_quotient'] = float(results['average_monthly_win']/results['average_monthly_loss'])
results['best_month'] = np.max(monthly_pnl.unstack())
results['worst_month'] = np.min(monthly_pnl.unstack())
results['max_dd'] = float(np.max(drawdowns(df)))
results['date_of_max_dd'] = np.argmax(drawdowns(df))
results['date_of_high'] = np.argmax(df[df.index < results['date_of_max_dd']]['Close'])
results['duration_of_dd'] = int((results['date_of_max_dd'] - results['date_of_high']).days)
results['distance_from_max_dd'] = int((pd.Timestamp('today')-results['date_of_max_dd']).days)
results['max_dd_1y'] = float(np.max(drawdowns(df[(pd.Timestamp('today')-pd.Timedelta('365D')).date():])))
results['date_of_max_dd_1y'] = np.argmax(drawdowns(df)[(pd.Timestamp('today')-pd.Timedelta('365D')).date():])

results['ytd_profit'] = performance_ytd(df)
results['3m_profit'] = performance_per_tf(df,90)
results['6m_profit'] = performance_per_tf(df,180)
results['12m_profit'] = performance_per_tf(df,360)

results['250d_vola'] = df.Close.rolling(250).std(ddof=0)[-1]/100
results['mar'] = MAR(df)
results['calmar'] = CAL_MAR(df)
results['sharpe_ratio'] = sharpe_ratio(df,3*365)
results['sterling_ratio'] = sterling_ratio(df,3*365)

def recovery_from_dd(df):
    ath_before_dd = np.max(df[df.index < results['date_of_max_dd_1y']]['Close'])
    date_of_ath = np.argmax(df[df.index < results['date_of_max_dd_1y']]['Close'])
    df_after_dd = df[df.index > results['date_of_max_dd_1y']]
    try:
        high_after_dd = df_after_dd[df_after_dd['Close']>ath_before_dd]['Close'][0]
        date_of_high = df_after_dd[df_after_dd['Close']>ath_before_dd].index[0]
    except IndexError:
        high_after_dd = np.NaN
        date_of_high = np.NaN
    
    
    if (high_after_dd > ath_before_dd):
        return (date_of_high - date_of_ath).days
    return np.NaN

results['recovery_days_dd_1y'] = recovery_from_dd(df)
results['description'] = open(working_folder + '\\templates\\system_descriptions\\' + system_name + '_description.txt').read()

facts = pd.DataFrame.from_dict(results,orient='index')
facts.columns = [system_name.replace('_',' ')]


#plot and safe the chart
def cm2inch(value):
    return value/2.54

height = 6.5
width = 11

jpg_file = working_folder + system_name + str(datetime.now().date()) + '.jpg'
fig = plt.figure()
plt.grid(color='#fce4d6')
plt.plot(df_to_plot.Close,color='#1b1b1e')
plt.ylabel('Price €')
plt.gcf().autofmt_xdate()
fig.set_size_inches(cm2inch(width),cm2inch(height))
plt.savefig(jpg_file,figsize=(0.1,0.1),bbox_inches='tight')


#copy template file and fill with data
template_file = working_folder + '\\templates\\exodus_portfolio_factsheet_template.xlsx'
filename = system_name + '_Factsheet_' + str(datetime.now().date()) + '.xlsx'
copy(template_file,working_folder + filename)


book = openpyxl.load_workbook(working_folder + filename)
writer = pd.ExcelWriter(working_folder + filename, engine='openpyxl') 
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

facts.to_excel(writer, 'facts')
monthly_pnl.to_excel(writer,'monthly_pnl')
df.to_excel(writer, 'rawdata')

FactSheet = book['FactSheet']
img = openpyxl.drawing.image.Image(working_folder + '\\templates\\exodus_logo.jpg')
FactSheet.add_image(img,'I1')

img = openpyxl.drawing.image.Image(jpg_file)
FactSheet.add_image(img,'B36')

yearly_pnl.to_excel(writer,'monthly_pnl',startrow=16)

writer.save()


#save a copy in the archive and clean up workspace
copy(working_folder + filename,working_folder + 'archive\\' + filename)
remove(working_folder + file_name)
remove(working_folder + raw_csv)
remove(jpg_file)