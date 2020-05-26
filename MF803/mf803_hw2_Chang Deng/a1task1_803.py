from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

def beta_calculator(X, y):
    df = pd.concat([X,y],axis=1).dropna()
    result = stats.linregress(df.iloc[:,0],df.iloc[:,1])
    beta = result.slope
    return beta

class ETF:

    def __init__(self,code:str,start_date:str,end_date:str):
        self.code = code
        self.start_date = start_date
        self.end_date = end_date

    def price_acquire(self):
        yf.pdr_override()
        data = pdr.get_data_yahoo(self.code, start=self.start_date, end=self.end_date)
        data = data.reset_index()
        self.data = data.loc[:, ['Date', 'Close']]
        return self

    def calculate_annualized_retrun(self):
        self.price_acquire()
        P0 = self.data.loc[0,'Close']
        Pt = self.data.loc[self.data.shape[0]-1,'Close']
        r = math.log(Pt/P0)
        self.annualized_retrun = r*252/self.data.shape[0]
        annual_return = {'Code': [self.code], 'Annualized_Return': [self.annualized_retrun]}
        return pd.DataFrame(annual_return)

    def calculate_return(self,frequency):
        self.price_acquire()
        data = self.data
        if frequency == 'Daily':
            n = 1
        if frequency == 'Monthly':
            n = 21
        data[''+self.code+'_'+frequency+'_return'] = data['Close'] / data['Close'].shift(n) - 1
        data = (data.dropna(axis=0, how='any')).reset_index(drop=True)
        data = data.drop(columns='Close')
        return data

    def calculate_standard_deviation(self):
        self.price_acquire()
        self.std = self.data['Close'].std(ddof=0)
        return self

    def benchmark_acquire(self,index_code = '^GSPC'):
        yf.pdr_override()
        data = pdr.get_data_yahoo(index_code, start=self.start_date, end=self.end_date)
        data = data.reset_index()
        self.benchmark = data.loc[:, ['Date', 'Close']]
        self.benchmark.rename(columns={'Close':'Index_Close'},inplace=True)
        return self

    def rolling_nday_corr(self,ndays = 90):
        self.price_acquire()
        self.benchmark_acquire('^GSPC')
        data = self.data
        benchmark = self.benchmark
        data['ETF_Daily_return'] = data['Close'] / data['Close'].shift(1) - 1
        benchmark['Index_Daily_return'] = benchmark['Index_Close'] / benchmark['Index_Close'].shift(1) - 1
        data = data.fillna(0)
        data = data.drop(columns=['Close'])
        benchmark = benchmark.fillna(0)
        benchmark = benchmark.drop(columns=['Index_Close'])
        data = pd.DataFrame.merge(data,benchmark,how='left')
        rolling_corr = data['ETF_Daily_return'].rolling(ndays).corr(data['Index_Daily_return'])
        rolling_corr = (rolling_corr.dropna(axis=0, how='any')).reset_index(drop=True)
        return rolling_corr

    def calculate_beta(self):
        self.price_acquire()
        self.benchmark_acquire('^GSPC')
        data = self.data
        benchmark = self.benchmark
        data['ETF_Daily_return'] = data['Close'] / data['Close'].shift(1) - 1
        benchmark['Index_Daily_return'] = benchmark['Index_Close'] / benchmark['Index_Close'].shift(1) - 1
        data = data.fillna(0)
        data = data.drop(columns=['Close'])
        benchmark = benchmark.fillna(0)
        benchmark = benchmark.drop(columns=['Index_Close'])
        data = pd.DataFrame.merge(data, benchmark, how='left')
        beta = beta_calculator(data['Index_Daily_return'], data['ETF_Daily_return'])
        return beta

    def rolling_nday_beta(self,ndays=90):
        self.price_acquire()
        self.benchmark_acquire('^GSPC')
        data = self.data
        benchmark = self.benchmark
        data['ETF_Daily_return'] = data['Close'] / data['Close'].shift(1) - 1
        benchmark['Index_Daily_return'] = benchmark['Index_Close'] / benchmark['Index_Close'].shift(1) - 1
        data = data.fillna(0)
        data = data.drop(columns=['Close'])
        benchmark = benchmark.fillna(0)
        benchmark = benchmark.drop(columns=['Index_Close'])
        data = pd.DataFrame.merge(data, benchmark, how='left')
        data = data.drop(columns=['Date'])
        rolling_beta = (data['Index_Daily_return'].rolling(ndays)).apply(lambda x: beta_calculator(x,data['ETF_Daily_return']), raw=False)
        rolling_beta = (rolling_beta.dropna(axis=0, how='any')).reset_index(drop=True)
        return rolling_beta

    def auto_regression(self):
        self.price_acquire()
        data = self.data
        data['ETF_Daily_return'] = data['Close'] / data['Close'].shift(1) - 1
        data = data.fillna(0).reset_index()
        data = data.drop(columns=['Date', 'Close'])
        data_lag = data.shift(1)
        data = pd.DataFrame.merge(data_lag, data, how='right')
        regress_result = stats.linregress(data.iloc[:, 0], data.iloc[:, 1])
        result = {'Code':self.code,'Alpha':[regress_result.slope],'P_Value':[regress_result.pvalue]}
        return pd.DataFrame(result)

def clean_data_examinate():
    code_list = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-13')
        etf.price_acquire()
        data = etf.data[etf.data.isnull().T.any()]
        print(data)
    return

def multi_annualized_return():
    etf = ETF('SPY','2010-01-01','2019-09-13')
    data = etf.calculate_annualized_retrun()
    etf.calculate_standard_deviation()
    data['Standard_Deviation'] = etf.std

    code_list = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    for code in code_list:
        etf = ETF(code,'2010-01-01','2019-09-13')
        result = etf.calculate_annualized_retrun()
        etf.calculate_standard_deviation()
        result['Standard_Deviation'] = etf.std
        data = data.append(result)

    return data

def multi_covariance(frequence = 'Daily'):
    etf = ETF('SPY','2010-01-01','2019-09-13')
    data = etf.calculate_return(frequence)
    code_list = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

    for code in code_list:
        etf = ETF(code,'2010-01-01','2019-09-13')
        r = etf.calculate_return(frequence)
        data = pd.DataFrame.merge(data,r,how='left',on='Date')
    corr = data.corr()
    cov = data.cov()
    return corr,cov

def draw_rolling_ndays_correlation(ndays=90):
    register_matplotlib_converters()
    code_list = ['SPY','XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-13')
        data = etf.rolling_nday_corr(ndays)
        X = (etf.data.loc[89:etf.data.shape[0], 'Date']).reset_index(drop=True)
        plt.plot(X,data)
        #plt.title(''+code+'_rolling_90days_corr')
    plt.show()

    return

def draw_capm_beta():
    register_matplotlib_converters()
    beta_list = []

    code_list = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-13')
        etf.price_acquire()
        X = (etf.data.loc[89:etf.data.shape[0], 'Date']).reset_index(drop=True)
        beta = etf.calculate_beta()
        beta_list = beta_list + [beta]
        data = etf.rolling_nday_beta()
        # plt.plot(X,data)
        # plt.title('' + code + '_rolling_90days_beta')
        # plt.show()
    list = {'Code':code_list,'Beta':beta_list}
    result = pd.DataFrame(list)
    return result

def multi_auto_regression():
    etf = ETF('SPY', '2010-01-01', '2019-09-13')
    data = etf.auto_regression()
    code_list = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-13')
        r = etf.auto_regression()
        data = data.append(r)

    return data


if __name__ == '__main__':
    #display all columns
    pd.set_option('display.max_columns', None)
    #display all rows
    pd.set_option('display.max_rows', None)
    # clean_data_examinate() #question a
    #annualized_return = multi_annualized_return()
    # print(annualized_return) #question b
    # daily_corr, daily_cov = multi_covariance('Daily')
    # monthly_corr, monthly_cov = multi_covariance('Monthly')
    # write = pd.ExcelWriter(r'C:\Users\DCrai\Desktop\collect.xlsx')
    # df1 = pd.DtaFrame(daily_corr)
    # df1.to_excel(write, sheet_name='daily_corr', index=False)
    # df2 = pd.DataFrame(monthly_corr)
    # df2.to_excel(write, sheet_name='monthly_corr', index=False)
    # df3 = pd.DataFrame(daily_cov)
    # df3.to_excel(write, sheet_name='daily_cov', index=False)
    # df4 = pd.DataFrame(monthly_cov)
    # df4.to_excel(write, sheet_name='monthly_cov', index=False)
    # write.save() #question c
    draw_rolling_ndays_correlation() #question d
    # b = draw_capm_beta() #question e
    # print(b)
    # ar = multi_auto_regression() #question f
    # print(ar)