import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta

# Function to transform date
def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    # 52 weeks 
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  
    return data

# Function to apply SVD
def svd_dept(train, n_components=8):
    
    # Initialize a dictionary to store reconstructed matrices for each department
    dept_reconstructed = {}
    
    train_data = train[['Dept', 'Store', 'Date', 'Weekly_Sales']]
    
    # Loop over each unique department
    for dept in train_data['Dept'].unique():
        # Filter data for the current department
        dept_data = train_data[train_data['Dept'] == dept]
        
        # Pivot the data to create a matrix X (stores as rows, weeks as columns)
        pivot_data = dept_data.pivot_table(index='Store', columns='Date', values='Weekly_Sales', fill_value=0)
        pivot_data = pivot_data.reset_index()
        
        # Center the data by subtracting each store's mean
        centered_data = pivot_data.drop(columns=['Store'], axis = 1)
        X = centered_data - centered_data.mean(axis=1).values.reshape(-1, 1)
        # Apply SVD to the centered data
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Retain only the top 8 components
        U_reduced = U[:, :n_components]
        S_reduced = np.diag(S[:n_components])
        Vt_reduced = Vt[:n_components, :]
        
        # Reconstruct the matrix using the top components
        X_reconstructed = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
        
        # Add back the store means to the reconstructed matrix
        X_reconstructed = X_reconstructed + pivot_data.mean(axis=1).values.reshape(-1, 1)
        store_column = pivot_data['Store'].values.reshape(-1, 1)
        X_reconstructed_with_store = np.hstack([store_column, X_reconstructed])

        # Store the reconstructed matrix in the dictionary with the department as the key
        dept_reconstructed[dept] = pd.DataFrame(X_reconstructed_with_store, index=pivot_data.index, columns=pivot_data.columns)
    
    return dept_reconstructed

# Function to update SVD tranformed Weekly_Sales into train data
def update_train(my_train, train):
    dept_updates = []
    for dept, dept_df in my_train.items():
        dept_long = dept_df.melt(id_vars=['Store'], var_name='Date', value_name='Weekly_Sales')
        dept_long['Dept'] = dept
        dept_updates.append(dept_long)
    dept_updates_df = pd.concat(dept_updates, ignore_index=True)

    train = train.merge(
        dept_updates_df,
        on=['Store', 'Dept', 'Date'],
        how='left',
        suffixes=('', '_new')
    )
    
    # Replace original 'Weekly_Sales' values with the new ones where available
    train['Weekly_Sales'] = train['Weekly_Sales_new'].combine_first(train['Weekly_Sales'])
    
    # Drop the temporary column used for the update
    return train.drop(columns='Weekly_Sales_new')

# Function to shift sales
def shift(test, threshold=1.1, shift=1):

    # Define the holiday period index for weeks 48-52
    idx = np.arange(48, 53) - 44  # Align weeks 48–52 with the start index used in test data
    baseline_row = test[test['Wk'].isin([48, 52])]
    surge_row = test[test['Wk'].isin([49, 50, 51])]
    holiday =  test[test['Wk'].isin([48, 49, 50, 51, 52])].copy()

    if len(baseline_row) != 0 and len(surge_row) != 0:
        baseline = baseline_row['Weekly_Pred'].sum() / len(baseline_row)
        surge = surge_row['Weekly_Pred'].sum() / len(surge_row)
        if np.isfinite(surge / baseline) and (surge / baseline > threshold):
            h = np.array(holiday['Weekly_Pred'])
            shifted_sale = ((7 - shift) / 7) * h
            shifted_sale[1:] = shifted_sale[1:] + (shift / 7) * h[:-1]
            shifted_sale[0] = h[0]
            test.loc[test['Wk'].isin([48, 49, 50, 51, 52]), 'Weekly_Pred'] = shifted_sale
    return test


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
my_train = svd_dept(train, n_components=8)
train = update_train(my_train, train)

test_depts = test['Dept'].unique()
for dept in test_depts:
    train_dept_data = train[train['Dept'] == dept]
    test_dept_data = test[test['Dept'] == dept]
        
    train_stores = train_dept_data['Store'].unique()
    test_stores = test_dept_data['Store'].unique()
    test_stores = list(set(train_stores).intersection(test_stores))
    for store in test_stores:
        tmp_train = train_dept_data[train_dept_data['Store'] == store].copy()
        tmp_train = preprocess(tmp_train)
        tmp_test = test_dept_data[test_dept_data['Store'] == store].copy()
        tmp_test = preprocess(tmp_test)

        # Create model matrices
        X_train = pd.get_dummies(tmp_train[['Yr', 'Wk']], drop_first=True)
        X_test = pd.get_dummies(tmp_test[['Yr', 'Wk']], drop_first=True)

        # Add the quadratic term for Yr
        X_train['Yr_squared'] = X_train['Yr'] ** 2
        X_test['Yr_squared'] = X_test['Yr'] ** 2
        xx = len(X_test)
        # Drop all constant (or all zero) columns from both X_train and X_test
        cols_to_drop = X_train.columns[X_train.std() == 0]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        if xx != len(X_test):
            print(dept, store)
        # Add the intercept column
        X_train.insert(0, 'const', 1)
        X_test.insert(0, 'const', 1)

        # Drop linear redundant columns
        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            
            # Extract the current column and all previous columns
            Y = X_train.iloc[:, i].values.astype(float)
            X = X_train.iloc[:, :i].values.astype(float)
            
            # Use numpy's lstsq to find the least squares solution
            coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

            # If the residuals are almost zero, the column can be represented as a linear combination of previous columns
            if np.sum(residuals) < 1e-16:
                cols_to_drop.append(col_name)
        
        # Drop those columns from both X_train and X_test
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        model = sm.OLS(tmp_train['Weekly_Sales'], X_train).fit()
        mycoef = model.params.fillna(0)
        
        tmp_pred = np.dot(X_test, mycoef)
        tmp_test['Weekly_Pred'] = tmp_pred
        
        is_holiday_weeks = tmp_test['Date'].between("2011-11-04", "2011-12-30").any()# 48 and 52

        if is_holiday_weeks:
            tmp_test = shift(tmp_test)
            
        tmp_test = tmp_test.drop(['Wk', 'Yr'], axis=1)
        
        if (dept, store) == (1, 1):
            tmp_test.to_csv('mypred.csv', index=False)
        else:
            tmp_test.to_csv('mypred.csv', index=False, header=False, mode='a')

pred = pd.read_csv('mypred.csv')
test = pd.read_csv('test.csv')
pred = pred.drop(columns=['IsHoliday'])
pred = pred.merge(test, on=['Date', 'Store', 'Dept'], how='right')
pred = pred.fillna(0)
pred.to_csv('mypred.csv', index=False)