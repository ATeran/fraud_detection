# Fraud Detection Case Study

## Objective
We have two objectives for this case study:
* Build a model to predict if a newly registered event is possibly fraudulent
* Create a dashboard for investigators to use which helps them identify new events that are worthy of investigation for fraud.  This will pull in new data regularly, and update a useful display for the investigation team

## Approach

### Model & App Development:
* Identify minimum set of features to fit a model
* Build a logistic regression model and a random forest classifier model and pickle the models
* Write a fraud prediction script that pulls new event data, predicts the probability the event is fraudulent (using the fit models), and push the results to a SQL database
* Setup a SQL database that inserts the event details and fraudulent probability for the event as a new record
* Launch a Flask app that queries the database and updates the fraud prediction dashboard, which displays the latest events (w/ fraud probabilities) and an aggregated count of Low, Med, High risk events
* Deploy the application to an EC2 instance on AWS

### Team Responsibilities:
* The entire team went through EDA together and decided on the features for the model
* Mike - developed / refined models
* Alan - wrote prediction script and connection to SQL database
* Elias - developed SQL database and deployed application on AWS
* Whitney - created Flask app

## EDA & Feature Engineering
Initial focus was to build a simple model without any feature engineering. We chose the following features for our model:
```
In [16]: fraud_df.columns
Out[16]: 
Index(['body_length', 'fb_published', 'has_analytics', 'has_header',
       'has_logo', 'name_length', 'org_facebook', 'org_twitter',
       'sale_duration', 'show_map', 'user_age', 'fraud'],
      dtype='object')
```
All of the columns above were numerical or already setup as dummies w/ 0 or 1 values.

We added a target 'fraud' variable and found that 1,293 events were tagged as fraudulent (out of 14,337 events).

<img src="https://github.com/whitneypenn/dsi-fraud-detection-case-study/blob/master/imgs/fraud_hist.png">

## Model & Results

We quickly built two models, a logistic regression model and a random forest model. The parameters for each model are as follows:
```python
log_reg_model = LogisticRegression(max_iter=5000, n_jobs=-1, verbose=2, class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=None, min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight="balanced")
```
Given that the objective is to prevent fraud, we determined that the recall metric was the most important as we want the highest true positive rate possible. False negatives are fraudulent events that the model missed, so we want to avoid those at all costs. Below are the results for our random forest and logistic regression models.

|Model | Accuracy| Recall | Precision
|------|---------|--------|-----------|
|Random Forest | 96.1% | 79.0% | 77.1%|
|Logistic Regression|78.0% | 87.3% | 27.4%|

The Logistic Regression model has the highest Recall, but lower accuracy and precision. The ROC curve shows both models are effective, with the Random Forest model beating out the Logistic Regression in terms of area under the curve.

<img src="https://github.com/whitneypenn/dsi-fraud-detection-case-study/blob/master/imgs/ROCcurv.png">

We also looked at the coefficients of our features in the logistic regression model to understand the impact of each feature.

|Coefficient | Feature |
|------|------------|
|-0.96 | org_twitter |
|-0.82 | user_age |
|-0.75 | sale_duration |
|-0.58 | org_facebook |
|-0.52 | has_analytics |
|-0.36 | body_length |
|-0.33 | has_header |
|-0.32 | fb_published |
|-0.24 | has_logo |
|-0.15 | name_length |
|-0.13 | show_map |

What's most interesting is that every feature decreases the log odds that the event is fraudulent. Intuitively, this makes sense, as the more details provided about an organiziation (number of fb followers, number of twitter followers, if the org has a logo, etc.) indicate the organization is likely not fake, and therefore the event is not fake either.

## Live Application Demo

### Overview of Steps:
* After fitting and pickling the model, we move to the live part of the application.
* The prediction script calls the live server to get a new event to predict (using our logistic regression model) then inserts the event record and prediction probability.
* The SQL database then stores the event record and predicted probability
* Then a flask app, built using a bootstrap template, queries the database for a select number of columns, and displays the results on the dashboard
* Lastly, the application is running live on an EC2 instance on AWS

<img src="https://github.com/whitneypenn/dsi-fraud-detection-case-study/blob/master/imgs/Screen%20Shot%202018-08-28%20at%203.07.04%20PM.png">

Now, we'll share a live demo...

## Future Work

* Engineer additional features including NLP on the name and description fields, and a few dummy features (e.g., currency, payout_type)
* Further tune the logistic regression and random forest models, and try a few others
* Add features to the web app (e.g., filtering, trends over time, etc.)
