import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

#%matplotlib

# f = open('data/data.json')
# data=json.loads(f.read())
# df = pd.DataFrame.from_dict(data)
# df['fraud'] = df['acct_type'].str.contains('fraud')
# X = df.select_dtypes(include=['bool','number'])
#
# scaler = StandardScaler()
# X = Xscaler.fit_transform(X)
# Y=df.fraud.astype(int)
# X = X.drop(columns=['fraud'])

fraud_df = pd.read_json('data/data.json')
# add fraud target column
fraud_mask = (fraud_df.loc[:, 'acct_type'] == 'fraudster_event') | (fraud_df.loc[:, 'acct_type'] == 'fraudster') | (fraud_df.loc[:, 'acct_type'] == 'fraudster_att')
fraud_df.loc[fraud_mask, 'fraud'] = 1
fraud_df['fraud'].fillna(0, inplace=True)
# drop columns for first run of model
cols_to_drop = ['acct_type', 'approx_payout_date', 'channels', 'country',
'currency', 'delivery_method', 'description', 'email_domain',
'event_created', 'event_end', 'event_published', 'event_start', 'name',
'object_id', 'org_desc', 'org_name',
'payee_name', 'payout_type', 'previous_payouts',
'sale_duration2', 'ticket_types','sale_duration',
'user_created', 'user_type', 'venue_address', 'venue_country', 'num_payouts', 'num_order',
'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'listed', 'gts']
fraud_df.drop(cols_to_drop, axis=1, inplace=True)
fraud_df.fillna(0, inplace=True)
X = fraud_df.drop('fraud', axis=1).values
y = fraud_df['fraud'].values




scaler = StandardScaler()
X = scaler.fit_transform(X)


model = LogisticRegression(max_iter=5000,
                           n_jobs=-1,
                           verbose=2,
                           class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=42)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Precision={precision:.3f}    Recall={recall:.3f}')
