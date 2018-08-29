from flask import Flask, request, flash, redirect, url_for, render_template
import requests
from werkzeug.utils import secure_filename
import numpy as np
import sys
import os
import pdb
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
from sklearn.datasets import load_iris

app = Flask(__name__)

# Pull from API and convert data to pandas dataframe

def get_fraud_df():
    conn = psycopg2.connect(database='frauddb')
    sql_query = '''
        SELECT
           sequence_number AS sequence_id,
           name as event_name,
           org_name,
           country,
           fraud_prob
        FROM
           fraud
    '''
    fraud_df = sqlio.read_sql_query(sql_query, conn)
    conn = None
    # fraud_df = fraud_df['fraud_prob'].astype(float)
    # fraud_df = fraud_df['sequence_id'].astype(float)
    return fraud_df

def define_risk_level(df):
    high_mask = df['fraud_prob'] >= .8
    med_mask = (df['fraud_prob'] >= .5) & (df['fraud_prob'] < .8)
    low_mask = df['fraud_prob'] < .5
    df.loc[high_mask, 'risk_level'] = 'High'
    df.loc[med_mask, 'risk_level'] = 'Medium'
    df.loc[low_mask, 'risk_level'] = 'Low'
    return df


@app.route('/')
def index():
    # data = pd.read_pickle('~/Downloads/record.json.list')
    # risk = np.random.rand(150,1)
    # data['fraud_prob'] = risk
    # data = define_risk_level(data)
    data = get_fraud_df()
    data = define_risk_level(data)
    return render_template('index.html', table=data)
    #return render_template('index.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
