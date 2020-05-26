import numpy as np
import pandas as pd
import pickle
import datetime
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Next day return
    stock_price = pd.read_excel('796 price-data.xlsx')
    stock_price = stock_price[stock_price.Date >= '2007-01-01']
    stock_price = stock_price.set_index('Date')
    stock_price = stock_price.fillna(method='bfill')  # bfill N/A
    return_next_date = stock_price.pct_change().shift(-1)
    return_next_date.dropna(inplace=True)
    # Macro
    macro = pd.read_excel('796 macro-data.xlsx')
    macro = macro[macro.Date >= '2006-01-01']
    macro = macro.set_index('Date')
    macro = macro.fillna(method='bfill')
    macro_lag = 12
    macro_monthly = macro.resample('M', convention='end').ffill()
    macro_data = np.zeros((return_next_date.shape[0], macro_lag + 1, macro.shape[1]))
    for j in range(0, macro_lag + 1):
        tmp = pd.DataFrame(index=macro.index)
        tmp = pd.DataFrame.merge(tmp, macro_monthly.shift(j), how='left', on='Date')
        tmp = tmp.fillna(method='bfill').loc[return_next_date.index, :]
        tmp = tmp.fillna(macro_monthly.shift(11).loc['2020-04-30'])
        macro_data[:, j, :] = tmp
    wb = pd.ExcelFile('796 micro-data.xlsx')
    sheets = wb.sheet_names
    micro_more = pd.read_excel('raw_data.xlsx', sheets='SQL Results')
    df = wb.parse(sheets[3])
    df = df.set_index('Date').loc['2007-01-04':'2020-04-17', :]
    return_data = return_next_date.values
    valid = [
        'S_FA_EXTRAORDINARY',
        'S_FA_OPERATEINCOME',
        'S_FA_INVESTINCOME',
        'S_STM_IS',
        'S_FA_FCFF',
        'S_FA_RETAINEDPS',
        'S_FA_CFPS',
        'S_FA_NETPROFITMARGIN',
        'S_FA_PROFITTOGR',
        'S_FA_ADMINEXPENSETOGR',
        'S_FA_IMPAIRTOGR_TTM',
        'S_FA_OPTOGR',
        'S_FA_ROE',
        'S_FA_ROA',
        'S_FA_ASSETSTOEQUITY',
        'S_FA_DEBTTOEQUITY',
        'S_FA_ASSETSTURN',
        'S_FA_OPTOEBT',
        'S_FA_OCFTOPROFIT',
        'S_FA_OPTODEBT'
    ]
    micro_data = np.zeros((df.shape[0], df.shape[1], len(sheets) + len(valid)))
    i = 0
    for s in sheets:
        df = wb.parse(s)
        df = df.set_index('Date').loc['2007-01-04':'2020-04-17', :]
        df = df.fillna(method='bfill')
        micro_data[:, :, i] = df.values
        i = i + 1
    j = len(sheets)
    for v in valid:
        mm = micro_more[['S_INFO_WINDCODE', 'REPORT_PERIOD', v]]
        mm['Date'] = [pd.to_datetime(str(int(dt))) for dt in mm['REPORT_PERIOD']]
        mmm = pd.pivot(mm, index='Date', columns='S_INFO_WINDCODE', values=v)
        date = pd.DataFrame(index=df.index)
        dd = pd.DataFrame.merge(date, mmm, on='Date', how='outer')
        dd = dd.sort_index()
        dd = dd.fillna(method='bfill')
        dd = dd.fillna(method='ffill')
        dd = pd.DataFrame.merge(date, dd, on='Date', how='left')
        micro_data[:, :, j] = dd.values
        j = j + 1
    for z in range(micro_data.shape[2]):
        print(np.isnan(micro_data[:, :, z]).sum())
    file = open('a1', 'wb')
    pickle.dump([macro_data, micro_data, return_data], file)
    file.close()