import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

np.random.seed(40)

def feature_engineering(train_x, one_hot_encoders={}):
    train_x = train_x.drop(['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude','PID','Exterior_1st',"Exterior_2nd",'Kitchen_AbvGr','Functional'],axis=1)
    winsorize_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    train_x = winsorize(train_x, winsorize_cols,0.95)

    #train_x['Garage_Yr_Blt'] = train_x['Garage_Yr_Blt'].fillna(0)
    # Use np.where() to calculate the age of the garage, or keep the missing values as NaN
    # Then fill the NaN values with 0
    train_x['Garage_Yr_Blt'] = np.where(train_x['Garage_Yr_Blt'].notna(), 2024 - train_x['Garage_Yr_Blt'], np.nan)
    train_x['Garage_Yr_Blt'] = train_x['Garage_Yr_Blt'].fillna(0)
    
    # Separate numerical and categorical data
    num_cols = train_x.select_dtypes(exclude=['object', 'uint8']).columns
    scaler = StandardScaler()
    train_x[num_cols] = scaler.fit_transform(train_x[num_cols])
    

    for categorical_col in train_x.select_dtypes(include=['object']).columns:
        # Get One-Hot
        if categorical_col in one_hot_encoders:
            one_hot_enc = one_hot_encoders[categorical_col]
            col_one_hot = one_hot_enc.transform(train_x[categorical_col].to_numpy().reshape(-1, 1)).toarray()
        else:
            one_hot_enc = OneHotEncoder(handle_unknown='ignore')
            col_one_hot = one_hot_enc.fit_transform(train_x[categorical_col].to_numpy().reshape(-1, 1)).toarray()
            one_hot_encoders[categorical_col] = one_hot_enc
        one_hot_df = pd.DataFrame(data=col_one_hot, columns=[f"{categorical_col}_{i}" for i in range(col_one_hot.shape[1])])
        train_x = pd.concat([train_x, one_hot_df], axis=1)
        train_x.drop([categorical_col], axis=1, inplace=True)
    
    return train_x, one_hot_encoders

    
def winsorize(df, col, quantile=0.95):
    M = df[col].quantile(quantile)  # Calculate 95% quantile
    df[col] = np.where(df[col] > M, M, df[col])  # Replace values > M with M
    return df


def feature_engineering_GBM(x, one_hot_encoders={}):
    x = x.drop(['PID'],axis=1)

    x['Garage_Yr_Blt'] = x['Garage_Yr_Blt'].fillna(0)

    for categorical_col in x.select_dtypes(include=['object']).columns:
        # Get One-Hot
        if categorical_col in one_hot_encoders:
            one_hot_enc = one_hot_encoders[categorical_col]
            col_one_hot = one_hot_enc.transform(x[categorical_col].to_numpy().reshape(-1, 1)).toarray()
        else:
            one_hot_enc = OneHotEncoder(handle_unknown='ignore')
            col_one_hot = one_hot_enc.fit_transform(x[categorical_col].to_numpy().reshape(-1, 1)).toarray()
            one_hot_encoders[categorical_col] = one_hot_enc
        one_hot_df = pd.DataFrame(data=col_one_hot, columns=[f"{categorical_col}_{i}" for i in range(col_one_hot.shape[1])])
        x = pd.concat([x, one_hot_df], axis=1)
        x.drop([categorical_col], axis=1, inplace=True)
    
    return x, one_hot_encoders 
    

def main():
    #print(f'Running fold {i+1}')
    #train_path = os.path.join('proj1', f'fold{i+1}', 'train.csv')
    #test_x_path = os.path.join('proj1', f'fold{i+1}', 'test.csv')

    train = pd.read_csv('train.csv')

    train_x = train.drop(["Sale_Price"], axis=1)
    train_y_enet = np.log(train["Sale_Price"])
    train_y_gbm = train["Sale_Price"]

    #test_y_path = os.path.join('proj1', f'fold{i+1}', 'test_y.csv')
    
    # Preprocessing
    train_x_enet, one_hot_encoders = feature_engineering(train_x)

    # Fit
    elastic_net = ElasticNetCV(l1_ratio=0.013376472, 
                            alphas=np.logspace(-4, 0, 50, endpoint=True), 
                            cv=80, 
                            random_state=21,
                            max_iter=10000)

    elastic_net.fit(train_x_enet, train_y_enet)
    
    # Submission
    test_x = pd.read_csv('test.csv')
    test_y = pd.read_csv('test_y.csv')
    test_y = test_y['Sale_Price']

    test_x_enet, _ = feature_engineering(test_x, one_hot_encoders)
    y_hat = elastic_net.predict(test_x_enet)

    submission = pd.DataFrame()
    submission['PID'] = test_x['PID']

    submission['Sale_Price'] = np.exp(y_hat)
    submission.to_csv("mysubmission1.txt", index=False)
    #mse_enet = np.sqrt(np.mean((np.log(test_y)-(y_hat))**2))
    #print(f'RMSE for elastic_net: {rmse_enet}')
    
    # Preprocessing
    train_x_GBM, e = feature_engineering_GBM(train_x)

    # Fit
    gbm = GradientBoostingRegressor(n_estimators=600, max_depth=5, subsample=0.5, learning_rate=0.05)
    gbm.fit(train_x_GBM, train_y_gbm)
    
    # Submission
    test_x_GBM, _ = feature_engineering_GBM(test_x, e)
    y_hat_GBM = gbm.predict(test_x_GBM)

    submission_GBM = pd.DataFrame()
    submission_GBM['PID'] = test_x['PID']

    submission_GBM['Sale_Price'] = y_hat_GBM
    submission_GBM.to_csv("mysubmission2.txt", index=False)
    #rmse_gbm = np.sqrt(np.mean((np.log(y_hat_GBM)-np.log(test_y))**2))
    #print(f'RMSE for gbm: {rmse_gbm}')
        
if __name__ == "__main__":
    main()