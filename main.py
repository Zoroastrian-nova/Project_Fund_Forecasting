
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
import sklearn.model_selection as model_selection
from functions import TabularDataset,MAELoss,run_models
from functions import tuning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import functions
import torch.optim as optim
import sklearn.metrics as metrics
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler

def main(tune_enabled=False):

    if torch.cuda.device_count() >=1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    data = pd.read_csv("./Data/data.csv")
    data = data[data["Beta_1"]!=0]
    data = data[(data["Volatility"].notna()) & (data["Volatility"]>1e-7)]
    data = data[(data["Alpha"]< data["Alpha"].quantile(q=0.99)) & (data["Alpha"]>data["Alpha"].quantile(q=0.01))]
    data.to_csv("./Data/data.csv",index = None)

    Alpha_norm = StandardScaler()
    data["Alpha"] = Alpha_norm.fit_transform(data["Alpha"].values.reshape(-1, 1))
    data = data.fillna(1e-7)



    label_name = ["Alpha"]
    features_name = ["Beta_1","Beta_2","Beta_3","Beta_4","Beta_5","Beta_6",
    "OpenPrice","HighPrice","LowPrice","ClosePrice",'ReturnAccumulativeNAV','Maturity','ManagementCostRatio',
    "TurnoverRate","CovertRate","MarketValue",'MaxDrawdown','ExpenseRatio', 'Flow', 'ROA', 'ROE',
    'Volatility','Skewness', 'Kurtosis', 'SharpeRatio', 'R^2', 't-stat_return','TransactionFee',
    'MOM_1', 'MOM_3','MOM_6', 'MOM_9', 'MOM_12', 'CummulativeNAVGrowth','Watermark', 'OperatingIncomeRatio','EarningPerShare',
    'RiskPremium1', 'SMB1', 'HML1', 'RMW1','CMA1', 'UMD1','TotalAsset', 
    'Return_3M', 'Return_6M', 'Return_9M','Return_12M']

    tensor_features = torch.tensor(data[features_name].values)
    tensor_label = torch.tensor(data[label_name].values)


    # Split tensors into training and validation sets
    X_train, X_val, y_train, y_val = model_selection.train_test_split(tensor_features, tensor_label, test_size=0.2, random_state=42)

    # Create datasets for training and validation sets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    # Create dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)


    # Import modules


    # Define model, optimizer, loss function, and metric function
    num_features = len(features_name)


    criterion = nn.MSELoss()
    metric = MAELoss()





    LR_config = {
        "Model":"LR",
        "input_size":num_features,
        "lr": tune.choice([1e-2,1e-3,1e-4,1e-5]),
        "step_size":tune.choice([1,5,10]),
        "gamma":tune.choice([0.1,0.25,0.5]),
        "max_epochs":tune.choice([50,100,150]),
        "max_patience":tune.choice([10,20,50]),
        "train_loader":train_loader,
        "val_loader":val_loader,
        }

    LSTM_config = {
        "Model":"LSTM",
        "input_size":num_features,
        "lr": tune.choice([1e-2,1e-3,1e-4,1e-5]),
        "step_size":tune.choice([1,5,10]),
        "gamma":tune.choice([0.1,0.25,0.5]),
        "max_epochs":tune.choice([50,100,150]),
        "max_patience":tune.choice([10,20,50]),
        "num_layers":tune.choice([2,4,8]),
        "hidden_size":tune.choice([32,64,128]),
        "train_loader":train_loader,
        "val_loader":val_loader,
        }

    Tfm_config = {
        "Model":"Transformer",
        "input_size":num_features,
        "lr": tune.choice([1e-6,1e-2,1e-3,1e-4,1e-5]),
        "step_size":5,
        "gamma":0.1,
        "max_epochs":250,
        "max_patience":20,
        "num_layers":tune.choice([2,4,6,8,16,24]),
        "hidden_size":tune.choice([128,256,512,1024]),
        "num_heads":tune.choice([1,2,3,6]),
        "dropout":0.2,
        "train_loader":train_loader,
        "val_loader":val_loader,
        }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=150,
        grace_period=10,
        reduction_factor=2
    )

    ray.init(_temp_dir='D:/Notes/Paper/temp_dir',
            runtime_env={
            #"py_modules":["functions"],
            "working_dir": "./", 
            "excludes" : ["/__pycache__/","/Data/","/Doc/","/logs/","/temp_dir/","/厦大经济论文模板20220802/"] # Replace with the path to your module
            },
            ignore_reinit_error=True)

    if tune_enabled:

        analysis = tune.run(
            tuning,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=Tfm_config,
            num_samples=30,
            #metric="train_loss",
            #mode="min",
            scheduler=scheduler,
            local_dir="./ray_results",
            verbose=1
        )

        df_tfm_result = analysis.dataframe(metric = "loss",mode = "min")
        df_tfm_result.to_csv("./tfm_ray_results.csv",index = None)


        analysis = tune.run(
            tuning,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=LSTM_config,
            num_samples=15,
            #metric="train_loss",
            #mode="min",
            scheduler=scheduler,
            local_dir="./ray_results",
            verbose=1
        )

        df_lstm_result = analysis.dataframe(metric = "loss",mode = "min")
        df_lstm_result.to_csv("./lstm_ray_results.csv",index = None)


        analysis = tune.run(
            tuning,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=LR_config,
            num_samples=10,
            #metric="train_loss",
            #mode="min",
            scheduler=scheduler,
            local_dir="./ray_results",
            verbose=1
        )

        df_lr_result = analysis.dataframe(metric = "loss",mode = "min")
        df_lr_result.to_csv("./lr_ray_results.csv",index = None)


    from functions import run_models
    Tfm_best_config = {
        "Model":"Transformer",
        "input_size":num_features,
        "lr": 1e-3,
        "step_size":5,
        "gamma":0.1,
        "max_epochs":150,
        "max_patience":20,
        "num_layers":8,
        "hidden_size":256,
        "num_heads":2,
        "dropout":0.2,
        "train_loader":train_loader,
        "val_loader":val_loader,
        }
    tfm_model = run_models(Tfm_best_config)



    LSTM_best_config = {
        "Model":"LSTM",
        "input_size":num_features,
        "lr": 0.01,
        "step_size":10,
        "gamma":0.1,
        "max_epochs":150,
        "max_patience":50,
        "num_layers":2,
        "hidden_size":64,
        "train_loader":train_loader,
        "val_loader":val_loader,
        }
    lstm_model = run_models(LSTM_best_config)


    from functions import run_models
    LR_best_config = {
        "Model":"LR",
        "input_size":num_features,
        "lr": 1e-2,
        "step_size":5,
        "gamma":0.5,
        "max_epochs":100,
        "max_patience":10,
        "train_loader":train_loader,
        "val_loader":val_loader,
        }
    lr_model = run_models(LR_best_config)


if __name__ == "__main__":
    main()