
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def get_beta(data):
    for id in data["Symbol"].unique():
        sample_data = data[data["Symbol"] == id]
        for date in np.sort(data["Date"].unique()):
            sample = sample_data[sample_data["Date"] <= date]
            if len(sample):
                ols = LinearRegression(fit_intercept=False)
                X,y = sample[["RiskPremium1","SMB1","HML1","RMW1","CMA1","UMD1"]].values,sample["ReturnDaily"].values
                mean = sample["ReturnDaily"].mean()
                volatility = sample["ReturnDaily"].std()
                skew = sample["ReturnDaily"].skew()
                kurtosis = sample["ReturnDaily"].kurtosis()
                SR = sample["ReturnDaily"].mean()/volatility
                t_stat_return = (sample["ReturnDaily"] - sample["ReturnDaily"].mean())/volatility
                t_stat_return = t_stat_return.values[-1]
                if sample["ReturnDaily"].min()<0:
                    max_drawdown = sample["ReturnDaily"].min()
                else:
                    max_drawdown = 0

                ols.fit(X,y)
                alpha = y[-1] - ols.predict(X)[-1]
                beta_list = [date,id,mean,volatility,skew,kurtosis,SR,ols.score(X,y),t_stat_return,max_drawdown,alpha].__add__(ols.coef_.tolist())

                yield beta_list

def get_factor(data):
    for id in data["Symbol"].unique():
        sample_data = data[data["Symbol"] == id]
        date = sample_data["Date"]
        symbol = [id for i in range(len(sample_data))]

        MOM_1 = sample_data["Alpha"].shift(1)
        MOM_3 = sample_data["Alpha"].shift(3)
        MOM_6 = sample_data["Alpha"].shift(6)
        MOM_9 = sample_data["Alpha"].shift(9)
        MOM_12 = sample_data["Alpha"].shift(12)

        Return_3M = sample_data["ClosePrice"]/sample_data["ClosePrice"].shift(3) - 1 
        Return_6M = sample_data["ClosePrice"]/sample_data["ClosePrice"].shift(6) - 1 
        Return_9M = sample_data["ClosePrice"]/sample_data["ClosePrice"].shift(9) - 1 
        Return_12M = sample_data["ClosePrice"]/sample_data["ClosePrice"].shift(12) - 1 
        

        factor_list = np.array((date,symbol,MOM_1,MOM_3,MOM_6,MOM_9,MOM_12,Return_3M,Return_6M,Return_9M,Return_12M))

        yield factor_list


def calculate_data():

    data = pd.read_csv("./Data/data_0.csv")
    data["Date"] = data["Trading_Date"]


    for symbol in data["Symbol"].unique():
        sample = data[data["Symbol"]==symbol].ffill()
        data[data["Symbol"]==symbol] = sample



    data = data[data["RiskPremium1"].notna()]
    data = data.fillna(value=1e-7)


    data["ReturnDaily"] = data["ReturnDaily"]*100
    data["Return"] = data["ReturnDaily"]
    data = data.dropna(axis=0)
    data[["RiskPremium1","SMB1","HML1","RMW1","CMA1","UMD1"]] = data[["RiskPremium1","SMB1","HML1","RMW1","CMA1","UMD1"]]*100
    data = data[(data["ReturnDaily"] < data["ReturnDaily"].quantile(0.99)) & (data["ReturnDaily"] > data["ReturnDaily"].quantile(0.01)) ]


    betas = pd.DataFrame([i for i in get_beta()],columns=["Date","Symbol","Mean","Volatility","Skewness","Kurtosis","SharpeRatio",
                                                        "R^2","t-stat_return","MaxDrawdown",
                                                        "Alpha","Beta_1","Beta_2","Beta_3","Beta_4","Beta_5","Beta_6"])
    data = pd.merge(left=data,right=betas,on=["Date","Symbol"],how="left")
    data = data[data['Alpha'].abs()>1e-7]


    factor_data = np.zeros([11,1])
 
    for factor in get_factor(data):

        factor_data = np.concatenate((factor_data,factor),axis=1)



    factor_data = factor_data.T
    factor_data = factor_data[1:,:]
    factor_data = pd.DataFrame(factor_data,columns=("Date","Symbol","MOM_1","MOM_3","MOM_6","MOM_9","MOM_12","Return_3M","Return_6M","Return_9M","Return_12M"))
    factor_data["Date"] = pd.to_datetime(factor_data["Date"])
    data["Date"] = pd.to_datetime(data["Date"])
    data = pd.merge(left=data,right=factor_data,on=["Date","Symbol"],how="left")
    data["ROA"] = data["NetProfit"]/data["TotalAsset"]
    data["ROE"] = data["NetProfit"]/data["TotalEquity"]
    data["ValueAdded"] = (data["Alpha"] + data["ExpenseRatio"]/12)*data["TotalAsset"].shift(1)
    data["Alpha"] = data["Alpha"].shift(-1)
    data[["MarketValue","TotalEquity","TotalLiability","TotalAsset"]] = 1e-8*data[["MarketValue","TotalEquity","TotalLiability","TotalAsset"]]
    data[["TotalRevenue","TransactionFee","TotalOperatingCost","TotalProfit","NetProfit","Flow"]] = 1e-6*data[["TotalRevenue","TransactionFee","TotalOperatingCost","TotalProfit","NetProfit","Flow"]]


    data.columns = ['Symbol', 'NAV', 'ReturnNAV', 'AccumulativeNAV',
        'ReturnAccumulativeNAV', 'AchieveReturn', 'MasterFundCode',
        'ReportTypeID_x', 'Startdate_x', 'EndDate', 'StateTypeCode', 'FundID_x',
        'TotalAsset', 'TotalLiability', 'TotalEquity', 'DATA_TYPE_ID',
        'TotalRevenue', 'TransactionFee', 'TotalOperatingCost', 'TotalProfit',
        'NetProfit', 'ExpenseRatio', 'Flow', 'ROA', 'ROE', 'FundClassID',
        'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Change',
        'ChangeRatio', 'TurnoverRate', 'CovertRate', 'MarketValue',
        'ReturnDaily', 'Trading_Date', 'RiskPremium1', 'SMB1', 'HML1', 'RMW1',
        'CMA1', 'UMD1', 'ReportDate', 'FundID', 'FundName', 'Satus', 'Type', 'EarningPerShare',
        'Watermark', 'OperatingIncomeRatio', 'OperatingCostRatio', 'ManagementCostRatio', 'TransactionCostRatio', 'OtherCostRatio',
        'TotalCostRatio', 'NAVReturn', 'OperatingPerformanceGrowth', 'NAVChangeRate', 'CummulativeNAVGrowth', 'TNAGrowth',
        'UnrealizedGrowth', 'StockReturn', 'DividendRate', 'FdCd', 'Status', 'InvStl', 'FoundDt',
        'StartDt', 'ExpiDt', 'Maturity', 'Date', 'Return', 'Mean', 'Volatility',
        'Skewness', 'Kurtosis', 'SharpeRatio', 'R^2', 't-stat_return',
        'MaxDrawdown', 'Alpha', 'Beta_1', 'Beta_2', 'Beta_3', 'Beta_4',
        'Beta_5', 'Beta_6', 'MOM_1', 'MOM_3', 'MOM_6', 'MOM_9', 'MOM_12',
        'Return_3M', 'Return_6M', 'Return_9M', 'Return_12M', 'ValueAdded']


    data.to_csv("./Data/data.csv",index=None)




