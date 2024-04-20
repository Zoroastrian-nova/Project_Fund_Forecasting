
import pandas as pd

def collect_data():
    balance = pd.read_excel('./Data/FUND_FIN_Balance.xlsx')
    income = pd.read_excel('./Data/FUND_FIN_Income.xlsx')


    fin_report = pd.merge(left= balance, right= income, how="inner",on=["MasterFundCode","EndDate"])
    fin_report = fin_report.drop(["ReportTypeID_y","FundID_y","Startdate_y"],axis = 1)
    fin_report["ExpenseRatio"] = fin_report["TotalOperatingCost"]/fin_report["TotalAsset"]
    fin_report["Flow"] = (fin_report["TotalAsset"] - (1+fin_report["ExpenseRatio"])*fin_report["TotalAsset"].shift(1))/fin_report["TotalAsset"].shift(1)
    fin_report["ROA"] = fin_report["NetProfit"]/fin_report["TotalAsset"]
    fin_report["ROE"] = fin_report["NetProfit"]/fin_report["TotalEquity"]


    NAV_1 = pd.read_excel('./Data/75AF2600.xlsx')
    NAV_2 = pd.read_excel('./Data/Fund_NAV_Month1.xlsx')
    NAV = pd.concat((NAV_1,NAV_2),axis=0)


    price = pd.read_excel('./Data/Fund_MKT_QuotationMonth.xlsx')
    data = NAV
    data = pd.merge(left=data,right = fin_report, how="left",left_on=["TradingDate","Symbol"],right_on=["EndDate","MasterFundCode"])
    data = pd.merge(left=data,right = price, how="inner",on=["TradingDate","Symbol"])


    factors = pd.read_excel("./Data/STK_MKT_FIVEFACMONTH.xlsx")
    factors["TradingMonth"] = pd.to_datetime(factors["TradingMonth"],format="mixed")
    factors.set_index("TradingMonth",inplace=True)



    factors = factors.resample("M").mean(numeric_only=True)
    data["Date"] = data["TradingDate"].copy( )
    data["Date"] = data["Date"].ffill()
    data["Date"] = pd.to_datetime(data["Date"])
    #data = pd.merge(left=data,right = factors, how="left",left_on="Date",right_on=factors.index)

    Df = pd.DataFrame([])

    for symbol in data["Symbol"].unique():
        sample = data[data["Symbol"]==symbol].ffill().copy()
        #sample["Date"] = pd.to_numeric(sample["Date"], errors='coerce')
        
        #sample["Date"] = pd.to_datetime(sample["Date"])
        #sample["Date"] = sample["Date"].dropna()
        sample.set_index("Date",inplace=True) 
        sample = sample.resample("M").mean(numeric_only=True)
        sample["Trading_Date"] = sample.index.copy()
        sample.index = range(len(sample))

        Df = pd.concat((Df,sample),axis=0)
        


    Df = pd.merge(left=Df,right = factors, how="left",left_on="Trading_Date",right_on=factors.index)


    fin_data_1 = pd.read_excel("./Data/RESSET_FDFININD_1.xls")
    fin_data_2 = pd.read_excel("./Data/RESSET_FDFININD_2.xls")
    fin_data_3 = pd.read_excel("./Data/RESSET_FDFININD_1(1).xls")
    fin_data_4 = pd.read_excel("./Data/RESSET_FDFININD_1 (2).xls")
    fin_data = pd.concat((fin_data_1,fin_data_2,fin_data_3,fin_data_4),axis=0)
    Df = Df[Df["Symbol"].notna()]
    Df["Symbol"] = [str(int(x)) for x in Df["Symbol"]]
    Df["Trading_Date"] = pd.to_datetime(Df["Trading_Date"])

    df = pd.merge(left=Df,right=fin_data,left_on=("Trading_Date","Symbol"),right_on=("报告日期","基金代码"),how="left")


    fund_info = pd.read_excel("./Data/RESSET_FDINFO_1.xls")
    df = pd.merge(left=df,right=fund_info,left_on=("Symbol"),right_on=("FdCd"),how="left")

    df["Maturity"] = df["Trading_Date"] - df["StartDt"]
    df["Maturity"] = [x.days/30 for x in df["Maturity"]]



    for symbol in df["Symbol"].unique():
        sample = df[df["Symbol"]==symbol].ffill()
        df[df["Symbol"]==symbol] = sample


    df = df.dropna(axis=0,thresh=20)
    #Df["TradingDate"] = data["TradingDate"]
    df.to_csv("./Data/data_0.csv",index=None)


