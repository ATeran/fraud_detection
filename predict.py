import pandas as pd
import numpy as np
import requests
import time
import psycopg2
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import pickle
import os 

if __name__ == '__main__':
    with open('log_reg_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)


    api_key = os.environ['SECRET_KEY']
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = 0

    live_updates = True
    while live_updates:
        response = requests.post(url, json={'api_key': api_key, 'sequence_number': sequence_number})
        raw_data = response.json()
        df = pd.DataFrame(raw_data['data'])
        cols_to_drop = ['channels', 'country', 'currency', 'delivery_method',
               'description', 'email_domain', 'event_created', 'event_end',
               'event_published', 'event_start',
              'listed', 'name', 'object_id',
               'org_desc', 'org_name', 'payee_name',
               'payout_type', 'previous_payouts',
               'ticket_types', 'user_created', 'user_type',
               'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude',
               'venue_name', 'venue_state']
        # Drop irrelevant columns
        df_alt = df.drop(cols_to_drop, axis=1)
        df_alt.fillna(0, inplace=True)
        X = np.array(df_alt)
        # Generate a prediction
        X = scaler.transform(X)
        prediction = model.predict(X)
        proba = model.predict_proba(X)
        df['fraud_prob'] = proba[0][1]
        # Insert into database
        cols_to_drop = ['ticket_types', 'previous_payouts']
        df = df.drop(cols_to_drop, axis=1)
        df['sequence_number']=raw_data['_next_sequence_number']

        conn = psycopg2.connect(dbname='frauddb', host='localhost',
                                user='ubuntu', password='ubuntu')
        engine = create_engine('postgresql://ubuntu:ubuntu@localhost:5432/frauddb')
        df.to_sql('fraud', engine, if_exists='append')

        conn.commit
        conn.close()
        time.sleep(600)



# cols_used = ['body_length',  'fb_published',  'org_facebook',  'org_twitter', 'show_map',
#               'user_age', 'has_analytics', 'has_header', 'name_length', 'has_logo',]
