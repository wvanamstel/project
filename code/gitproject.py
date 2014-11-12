import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import subprocess
import requests
import json
import time
import csv
from itertools import chain
from sklearn.svm import OneClassSVM, SVC
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score


def plot_pull_requests_merged(df, users):
    #plots weekly merged pull requests
    temp = pd.DataFrame()
    fig = plt.figure()
    temp['timestamp'] = df[df.a_actor_attributes_login==usr].a_created_at
    temp['merged_request'] = df[df.a_actor_attributes_login==usr].a_payload_pull_request_merged
    temp.sort('timestamp', inplace=True)

    #construct a time series object
    ts = pd.Series(temp.merged_request.values, index=temp.timestamp)
    ts_res = ts.resample('W', how='sum')
    ts_res.plot()

    return None


def get_followers(user, pwd):
    base_url = 'https://api.github.com/users/'
    url = base_url + user
    r = requests.get(url, auth=('wvanamstel',pwd))
    #assert r.status_code == 200
    return r.json()['followers'], r.json()['public_repos']


def get_repos(user):
    base_url = 'https://api.github.com/users/'
    url = base_url + user + '/repos'
    r = requests.get(url)
    assert r.status_code == 200
    return r.json

def store_super_users(pwd):
    top_starred = pd.read_csv('../data/top_projects.csv')
    top_projects = top_starred.repository_name.unique()[:1000]

    data = []
    columns = ['repository', 'user1', 'user2']
    data.append(columns)
    #get top project contributors
    for i in range(len(top_projects)):
        print i
        project = top_projects[i]
        owner = top_starred[top_starred['repository_name']==project].repository_owner.unique()[0]
        url = 'https://api.github.com/repos/' + owner + '/' + project + '/stats/contributors'
        r = requests.get(url, auth=('wvanamstel', pwd))
        time.sleep(1)   #don't overwhelm the api
        records = len(r.json())

        #require that a repo has at least 5 contributors to be included
        if records > 5:
            temp = []
            temp.append(project)
            for i in xrange(1,3):
                try:
                    temp.append(r.json()[records - i]['author']['login'])
                except TypeError:
                    continue
            data.append(temp)

    #write super users to csv file
    write_to_csv('top_users.csv', data)
   
    return None


def get_github_user_data(df, pwd):
    ''''
    access github api to scrape user data
    '''
    top_users = df.user1.values
    second_users = df.user2.values
    all_users = np.hstack((top_users, second_users))

    #get the column names, which are the keys of the json dict
    rndm = requests.get('https://api.github.com/users/mdo', auth=('wvanamstel', pwd))
    cols = rndm.json().keys()
    data = []
    tmp = [['user']]
    tmp.append(cols)
    tmp = list(chain.from_iterable(tmp))
    data.append(tmp)

    for i in xrange(all_users.shape[0]):
        if (i % 100 == 0):
            print i
        tmp = [[all_users[i]]]
        url = 'https://api.github.com/users/' + str(all_users[i])
        user_data = requests.get(url, auth=('wvanamstel', pwd))
        tmp.append(user_data.json().values())
        tmp = list(chain.from_iterable(tmp))
        for j in range(len(tmp)):
            try:
                tmp[j] = str(tmp[j])
            except UnicodeError:
                tmp[j] = ''
    
        data.append(tmp)

    #write to csv
    write_to_csv('user_details.csv', data)
  
    return None

def get_user_events(df, pwd):
    '''
    access github api to scrape user event data
    '''
    #top_users = df.user1.values
    #second_users = df.user2.values
    all_users = df.values #np.hstack((top_users, second_users))

    #get data
    data = [['user', 'repo', 'event_type', 'action', 'timestamp', 'public']]
    for i in xrange(all_users.shape[0]):
        if (i % 10 == 0):
            print i
        if (i % 100 == 0):
            write_to_csv('user_events' + str(i/100) + '.csv', data)
            data = [['user', 'repo', 'event_type', 'action', 'timestamp', 'public']]

        base_url = 'https://api.github.com/users/' + str(all_users[i]) + '/events?page='
        for j in xrange(1,11):
            url = base_url + str(j)
            try:
                event_data = requests.get(url, auth=('wvanamstel', pwd))
            except requests.exceptions.ConnectionError as e:
                print e
                time.sleep(10)
                continue
            if event_data.status_code == 200:
                for k in range(len(event_data.json())):
                    temp = []
                    temp.append(event_data.json()[k]['actor']['login'])
                    temp.append(event_data.json()[k]['repo'])
                    temp.append(event_data.json()[k]['type'])
                    try:
                        temp.append(event_data.json()[k]['payload']['action'])
                    except KeyError:
                        temp.append('NA')
                    temp.append(event_data.json()[k]['created_at'])
                    temp.append(event_data.json()[k]['public'])
                    data.append(temp)

    #write to csv
    write_to_csv('user_events_last.csv', data)

    return None

def write_to_csv(filename, data):
    fout = open(filename, 'wb')
    a = csv.writer(fout)
    a.writerows(data)
    fout.close()

def stitch_together():
    #combine user details to 1 csv file
    df1 = pd.read_csv('../data/raw/top_user_details.csv')
    df2 = pd.read_csv('../data/raw/top_user_details2.csv')
    out = pd.concat([df1, df2], axis=0)
    out.to_csv('../data/user_details_all.csv')

    #combine event files into 1
    df = pd.read_csv('./user_events1.csv')
    for i in range(2,5):
        filename = './user_events' + str(i) + '.csv'
        df_temp = pd.read_csv(filename)
        df = pd.concat([df, df_temp], axis=0)

    df_last = pd.read_csv('./user_events_last.csv')
    df = pd.concat([df, df_last])
    df.to_csv('../data/user_events.csv')

    return None

def load_data(fin_users, fin_events):
    df_users = pd.read_csv(fin_users)
    df_events = pd.read_csv(fin_events)

    #clean up data frames
    df_users = df_users[df_users.public_repos!='False']
    df_users = df_users[df_users.site_admin!='Not Found']

    #clean up user event data
    df_events.drop('Unnamed: 0', axis=1, inplace=True)
    df_events.drop('public', axis=1, inplace=True)
    #clean up repo column
    temp = df_events['repo'].apply(lambda x: x.rsplit()[-1].rstrip('\'}'))
    df_events.repo = temp.apply(lambda x: x[x.find('/') + 1:])
    df_events.timestamp = pd.to_datetime(df_events.timestamp)   #convert to date time

    if ('TeamAddEvent' in df_events.columns):
        df_events.drop('TeamAddEvent', axis=1, inplace=True)
  
    #get daily averages of events
    df_events = bucket_events(df_events)

    return df_users, df_events

def bucket_events(df, freq='d'):
    #make dummy variables from the eventtype column
    dums = pd.get_dummies(df.event_type)
    cols = dums.columns
    new = pd.concat((df, dums), axis=1)
    new = new.set_index(new.timestamp.values)
    #preserve the user column
    new = pd.concat((new.iloc[:,0], new.iloc[:,5:]), axis=1)

    #get the frequency of events per time period (default=daily)
    #compute the average daily event frequency
    bucket_average = pd.DataFrame()#columns=cols)
    for user in new.user.unique():
        temp = new[new.user==user]
        temp2 = pd.DataFrame(np.mean(temp.resample(freq, how='mean'))).transpose()
        temp2['user'] = user
        bucket_average= pd.concat((bucket_average, temp2), axis=0)

    return bucket_average

    
def fit_model(df_to_fit, df_to_predict):  #for prelim testing purposes
    #read data
    X = df_to_fit.values

    #scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rs = ShuffleSplit(X.shape[0], n_iter = 1, random_state=31)

    for train_ind, test_ind in rs:
        train = X[train_ind]
        test = X[test_ind]

    #clf = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.001)
    clf = SVC()
    clf.fit(train)
    #predict
    pred_test = clf.predict(test) 
    true =[1] * pred_test.shape[0]
    f1 = f1_score(true, pred_test)
    print f1
    ac = accuracy_score(true, pred_test)
    print ac

    X_to_pred = df_to_predict.values
    X_to_pred = scaler.fit_transform(X_to_pred)
    predictions = clf.predict(X_to_pred)

    return predictions

def find_params(df):
    '''
    manual gridsearch for the oneclasssvm classifier, as GridSearchCV is
    giving unexpected results: rbf kernel with nu=0.001, gamma=0.005
    '''
    X = df.values

    #scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rs = ShuffleSplit(X.shape[0], n_iter = 1, random_state=31)

    for train_ind, test_ind in rs:
        train = X[train_ind]
        test = X[test_ind]

    kernel = ['rbf', 'poly']
    nu = [0.3, 0.5, 0.7]
    gamma = [0.01, 0.05, 0.15, 0.25]
    degree= [2,3]
    coef0 = [0.05, 0.1]
    nu_tune = list(np.arange(0.01, 0.25, 0.01))
    gamma_tune = list(np.arange(0.005, 0.05,0.005))

    for kern in kernel:   
        f1_final = 0
        if kern=='rbf':
            for n in nu_tune:
                for g in gamma_tune:
                    clf = OneClassSVM(kernel=kern, gamma=g, nu=n)
                    clf.fit(train)
                    pred = clf.predict(test)
                    true =[1] * pred.shape[0]
                    f1 = f1_score(true, pred)
                    ac = accuracy_score(true, pred)
                    if f1 > f1_final:
                        f1_final = f1
                        nu_final = n
                        gamma_final= g
            print 'rbf'
            print 'f1: ', f1_final
            print 'nu: ', nu_final
            print 'gamma: ', gamma_final
        else:
            for coef in coef0:
                for deg in degree:
                    clf = OneClassSVM(kernel=kern, degree=deg, coef0=coef)
                    clf.fit(train)
                    pred = clf.predict(test)
                    true =[1] * pred.shape[0]
                    f1 = f1_score(true, pred)
                    ac = accuracy_score(true, pred)
                    if f1 > f1_final:
                        f1_final = f1
                        nu_final = n
                        gamma_final= g
            print 'poly'
            print 'f1: ', f1_final
            print 'nu: ', nu_final
            print 'gamma: ', gamma_final

    return None


if __name__ == '__main__':
    #connect_to_mongoDB()
    #df_experts = read_data('data/experts.csv')
    #features = build_features(df_experts)
    print 'Loading data'
    df_user, df_events = load_data('../data/top_user_details_all.csv', '../data/top_user_events.csv')
    df_user_pred, df_events_pred = load_data('../data/user_details.csv', '../data/user_events.csv')
    cols = ['user', 'public_repos', 'followers','following','public_gists']
    df_small = df_user[cols]
    df_in = pd.merge(df_small, df_events, on='user')
    df_in = df_in.drop_duplicates()
    df_in = df_in.iloc[:,1:]    #drop user names

    print 'Fitting model'
    fit_model(df_in)