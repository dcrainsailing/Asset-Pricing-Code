from a1task1_803 import *
import statsmodels.api as sm
import numpy as np
import pandas as pd
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
import datetime

class factor_return():

    def __init__(self,start_date:str,end_date:str):
        self.start_date = start_date
        self.end_date = end_date

    def price_acquire(self):
        ff = pd.read_csv(r'F-F_Research_Data_Factors_daily.csv', header=3)
        ff = ff.dropna(axis=0, how='any')
        ff.rename(columns={'Unnamed: 0': 'Date','Mkt-RF':'Mkt_RF'}, inplace=True)  # "Date" --str类型
        ff = ff.drop(columns='RF')
        ff = ff[(ff['Date'] > (self.start_date).replace('-','')) & (ff['Date'] < (self.end_date).replace('-',''))]
        self.data = ff.reset_index(drop=True)
        return self

    def rolling_nday_corr(self,ndays = 90):
        register_matplotlib_converters()

        self.price_acquire()
        data = self.data
        #print(data.describe())
        X = (data.loc[ndays - 1:, 'Date']).reset_index(drop=True)
        X = X.apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d"))

        i = 1
        while i < data.shape[1]:
            j = 1
            while j < data.shape[1] -i:
                rolling_corr = data.iloc[:,i].rolling(ndays).corr(data.iloc[:,i+j])
                rolling_corr = (rolling_corr.dropna(axis=0, how='any')).reset_index(drop=True)
                plt.plot(X,rolling_corr)
                plt.title('Correlation of '+data.columns[i]+' and '+data.columns[i+j]+'')
                plt.show()
                j = j + 1
            i = i + 1
        fig=plt.figure(figsize=(100,100))

        return

    def examine_distribution(self):
        self.price_acquire()
        data = self.data
        i = 1
        while i < data.shape[1]:
            sm.qqplot(data.iloc[:,i],fit=True,line='45')
            plt.title('Normality test of '+data.columns[i]+' factor')
            plt.show()
            i = i + 1
        return

def fama_french_model():
    code_list = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    beta1 = []
    beta2 = []
    beta3 = []
    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-14')
        etf.price_acquire()
        etf.data['ETF_Daily_return'] = (etf.data['Close'] / etf.data['Close'].shift(1) - 1)
        etf.data['Date'] = etf.data['Date'].apply(lambda x: x.strftime("%Y%m%d"))
        data = pd.DataFrame.merge(etf.data, ff.data, how='left', on='Date')
        data = data.dropna(axis=0, how='any')
        model = OLS(y=data.ETF_Daily_return, x=data[['Mkt_RF','SMB','HML']])
        beta1 = beta1 + [model.beta[0]]
        beta2 = beta2 + [model.beta[1]]
        beta3 = beta3 + [model.beta[2]]
    result = {'Code':code_list,'Beta_Mkt_RF':beta1, 'Beta_SMB':beta2, 'Beta_HML':beta3}
    result = pd.DataFrame(result)
    return result

def rolling_ndays_ffmodels(ndays=90):
    register_matplotlib_converters()
    code_list = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-14')
        etf.price_acquire()
        etf.data['ETF_Daily_return'] = (etf.data['Close'] / etf.data['Close'].shift(1) - 1)
        etf.data['Date'] = etf.data['Date'].apply(lambda x: x.strftime("%Y%m%d"))
        data = pd.DataFrame.merge(etf.data, ff.data, how='left', on='Date')
        data = data.dropna(axis=0, how='any')
        model = PandasRollingOLS(y=data.ETF_Daily_return, x=data[['Mkt_RF','SMB','HML']], window=ndays)
        X = (data.loc[ndays:, 'Date']).reset_index(drop=True)
        X = X.apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
        plt.plot(X, model.beta)
        plt.title('Beta to the Fama-French factors of ETF:'+ code +'')
        plt.show()
    return

def residual():
    residual_mean = []
    residual_std = []
    resid = pd.DataFrame()
    resid_lag = pd.DataFrame()
    auto_alpha = []
    auto_pvalue = []

    code_list = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    for code in code_list:
        etf = ETF(code, '2010-01-01', '2019-09-14')
        etf.price_acquire()
        etf.data['ETF_Daily_return'] = (etf.data['Close'] / etf.data['Close'].shift(1) - 1)
        etf.data['Date'] = etf.data['Date'].apply(lambda x: x.strftime("%Y%m%d"))
        data = pd.DataFrame.merge(etf.data, ff.data, how='left', on='Date')
        data = data.dropna(axis=0, how='any')
        model = OLS(y=data.ETF_Daily_return, x=data[['Mkt_RF', 'SMB', 'HML']])
        resid['' + code + '_resids'] = model.resids
        sm.qqplot(resid['' + code + '_resids'], fit=True, line='45')
        plt.title('Normality test of daily residuals for ETF:' + code + '')
        plt.show()
        residual_mean = residual_mean + [np.mean(resid['' + code + '_resids'])]
        residual_std = residual_std + [np.std(resid['' + code + '_resids'])]
        resid_lag['' + code + '_resids_lag'] = resid['' + code + '_resids'].shift(1)
        residual = pd.concat([resid_lag['' + code + '_resids_lag'], resid['' + code + '_resids']], axis=1).dropna()
        regress_result = stats.linregress(residual.iloc[:, 0], residual.iloc[:, 1])
        auto_alpha = auto_alpha + [regress_result.slope]
        auto_pvalue = auto_pvalue + [regress_result.pvalue]

    result = {'Code': code_list, 'E_Mean': residual_mean, 'E_std': residual_std}
    result = pd.DataFrame(result)

    auto = {'Code': code_list, 'Alpha': auto_alpha, 'P_Value': auto_pvalue}
    auto = pd.DataFrame(auto)

    return result, auto

if __name__ == '__main__':
    # display all columns
    pd.set_option('display.max_columns', None)
    # display all rows
    pd.set_option('display.max_rows', None)
    # question a
    ff = factor_return('1926-07-01', '2019-07-31')
    ff.price_acquire()
    #print(ff.data.describe())
    # question b
    #corr = ff.data.corr()
    #cov = ff.data.cov()
    #print(corr)
    #print(cov)
    # question c
    #ff.rolling_nday_corr(90)
    #ff.examine_distribution()
    #question d
    #print(fama_french_model())
    # question e
    #rolling_ndays_ffmodels()
    # question f
    # norm, auto = residual()
    # print(norm)
    # print(auto)