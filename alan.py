import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold as skf
# df = pd.read_json('data.json')
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# Cross Validation


if __name__ == '__main__':
    fraud_df = pd.read_json('data/data.json')
    # add fraud target column
    fraud_mask = (fraud_df.loc[:, 'acct_type'] == 'fraudster_event') | (fraud_df.loc[:, 'acct_type'] == 'fraudster') | (fraud_df.loc[:, 'acct_type'] == 'fraudster_att')
    fraud_df.loc[fraud_mask, 'fraud'] = 1
    fraud_df['fraud'].fillna(0, inplace=True)
    # drop columns for first run of model
    cols_to_drop = ['acct_type', 'approx_payout_date', 'channels', 'country', 'currency', 'delivery_method', 'description', 'email_domain',
    'event_created', 'event_end', 'event_published', 'event_start', 'name',
    'object_id', 'org_desc', 'org_name',
    'payee_name', 'payout_type', 'previous_payouts',
    'sale_duration2', 'ticket_types',
    'user_created', 'user_type', 'venue_address', 'venue_country',
    'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'listed', 'gts', 'num_payouts','num_order','sale_duration']
    fraud_df.drop(cols_to_drop, axis=1, inplace=True)
    fraud_df.fillna(0, inplace=True)
    X = fraud_df.drop('fraud', axis=1).values
    y = fraud_df['fraud'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features="auto",
                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, bootstrap=True, oob_score=False,
                                n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                class_weight="balanced")

    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    acc = rf.score(X_test, y_test)


    print('Random Forest Accuracy is: {0:.3f}'.format(accuracy_score(rf_preds, y_test)))
    print('Random Forest Recall is: {0:.3f}'.format(recall_score(rf_preds, y_test)))
    print('Random Forest Precision is: {0:.3f}'.format(precision_score(rf_preds, y_test)))



# BASELINE:
# Random Forest Accuracy is: 0.931
# Random Forest Recall is: 0.660
# Random Forest Precision is: 0.480
