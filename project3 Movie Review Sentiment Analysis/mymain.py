import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(f'Process split {split+1}')

embedding_columns = [f'embedding_{i}' for i in range(1, 1537)]

X_train = train[embedding_columns].values
y_train = train['sentiment']

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 50)

elastic_net = ElasticNetCV(l1_ratio = 0.01, 
                            alphas = [0.0001], 
                            cv = cv, random_state = 50,
                            max_iter = 10000)

elastic_net.fit(X_train, y_train)


X_test = test[embedding_columns].values


# Predict probabilities
y_pred_probs = elastic_net.predict(X_test)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test['id'],          
    'prob': y_pred_probs
})

# Save to CSV
submission.to_csv('mysubmission.csv', index=False)