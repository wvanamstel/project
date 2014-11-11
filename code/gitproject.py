import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pymongo import MongoClient
from pymongo import errors
import subprocess
import requests
import json
import time
import csv
from itertools import chain
from sklearn.svm import OneClassSVM
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


def connect_to_mongoDB():
    #set up SSH tunnel to remote server
    subprocess.Popen('ssh -L 27017:dutihr.st.ewi.tudelft.nl:27017 ghtorrent@dutihr.st.ewi.tudelft.nl', 
                     shell=True, close_fds=True)

    #connect to remote mongodb
    client = MongoClient("localhost", 27017)
    client.github.authenticate('ghtorrentro', 'ghtorrentro')
    db = client.github

    #connect to collection 'issues'
    collection = db.issues
    collection.find_one()


def read_data(filename, explore=False):
    #read in data from csv file, exported github archive sql query
    df = pd.read_csv(filename)

    #take out the bots
    bots = ['opencv-pushbot', 'openshift-bot']
    for bot in bots:
        df = df[df.a_actor_attributes_login!=bot]

    #convert date/time cols into a datetime object
    df.a_created_at =  pd.to_datetime(df.a_created_at)
    df.a_repository_created_at = pd.to_datetime(df.a_repository_created_at)

    #convert True/False and NA ---> 1 and 0
    df['a_payload_pull_request_merged'] = df['a_payload_pull_request_merged'].apply(lambda x: 1 if x==True else 0)
    df['a_payload_commit_flag'] = df['a_payload_commit_flag'].apply(lambda x: 1 if x==True else 0)

    #print out some properties of the dataframe
    if explore:
        print 'Data types: \n', df.dtypes, '\n\n\n'
        print 'Shape: ', df.shape
        print 'User info: '
        for usr in df_experts.a_actor_attributes_login.unique():
            df_user = df_experts[df_experts.a_actor_attributes_login==usr]
            print usr, df_user.shape, df_user.a_payload_pull_request_merged.sum()

    return df


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


def build_features(df, pwd):
    #select and engineer features
    users = df.a_actor_attributes_login.unique()
    metrics={}
    for usr in users:
        #prep data
        temp = df[df.a_actor_attributes_login==usr]
        temp.sort('a_created_at', inplace=True)
        temp = temp.set_index('a_created_at')

        #get features
        weekly_pulls = np.mean(temp['a_payload_pull_request_merged'].resample('W', how='sum'))
        weekly_commits = np.mean(temp['a_payload_commit_flag'].resample('W', how='sum'))
        lang_list = [x for x in list(temp['a_repository_language'].unique()) if str(x)!='nan']

        #urls and cumul number of unique repos owned or forked by user
        repos_url = temp[temp['a_repository_owner'] == usr].a_repository_url.unique()
        repos_url = repos_url.shape[0]

        metrics[usr] = {}
        metrics[usr]['weekly_pulls'] = weekly_pulls
        metrics[usr]['weekly_commits'] = weekly_commits
        metrics[usr]['languages'] = lang_list
        metrics[usr]['num_followers'] = num_followers
        metrics[usr]['num_repos'] = num_repos

    return metrics


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
        if (i % 50 == 0):
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
    write_to_csv('top_users_details.csv', data)
  
    return None

def get_user_events(df, pwd):
    top_users = df.user1.values
    second_users = df.user2.values
    all_users = np.hstack((top_users, second_users))

    #get data
    data = [['user', 'repo', 'event_type', 'action', 'timestamp', 'public']]
    for i in xrange(601, all_users.shape[0]):
        if (i % 5 == 0):
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
    write_to_csv('top_users_events_last.csv', data)

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
    out.to_csv('../data/top_user_details_all.csv')

    #combine event files into 1
    df = pd.read_csv('../data/raw/user_events1.csv')
    for i in range(1,13):
        filename = '../data/raw/user_events' + str(i) + '.csv'
        df_temp = pd.read_csv(filename)
        df = pd.concat([df, df_temp], axis=0)

    df_last = pd.read_csv('../data/raw/top_users_events_last.csv')
    df = pd.concat([df, df_last])
    df.to_csv('../data/top_user_events.csv')

    return None

def load_data():
    df_users = pd.read_csv('../data/top_user_details_all.csv')
    df_events = pd.read_csv('../data/top_user_events.csv')

    #clean up data frames
    df_users = df_users[df_users.user != 'nurupu'] #empty user record
    #shif columns of certain users (about 30) who are missing col values
    ind_to_shift = df_users[df_users.public_repos=='False'].index
    df_temp = df_users.iloc[ind_to_shift]
    df_temp = pd.concat([df_temp.iloc[:,:2], df_temp.iloc[:,2:].shift(1, axis=1)], axis=1)
    df_temp = pd.concat([df_temp.iloc[:,:5], df_temp.iloc[:,5:].shift(1, axis=1)], axis=1)
    df_temp = pd.concat([df_temp.iloc[:,:10], df_temp.iloc[:,10:].shift(1, axis=1)], axis=1)
    df_temp = pd.concat([df_temp.iloc[:,:12], df_temp.iloc[:,12:].shift(1, axis=1)], axis=1)
    df_temp = pd.concat([df_temp.iloc[:,:14], df_temp.iloc[:,14:].shift(1, axis=1)], axis=1)
    df_temp = pd.concat([df_temp.iloc[:,:17], df_temp.iloc[:,17:].shift(1, axis=1)], axis=1)
    df_temp.public_repos = 0
    df_temp.following = 0
    df_temp.iloc[:,23], df_temp.iloc[:,24] = df_temp.iloc[:,24].values, df_temp.iloc[:,23].values
    df_users.iloc[ind_to_shift] = df_temp.values

    df_users.public_repos = df_users.public_repos.astype(int)
    df_users.followers = df_users.followers.astype(int)
    df_users.public_gists = df_users.public_gists.astype(int)
    df_users.drop('Unnamed: 0', axis=1, inplace=True)

    df_events.drop('Unnamed: 0', axis=1, inplace=True)
    df_events.drop('public', axis=1, inplace=True)
    #clean up repo column
    temp = df_events['repo'].apply(lambda x: x.rsplit()[-1].rstrip('\'}'))
    df_events.repo = temp.apply(lambda x: x[x.find('/') + 1:])
    df_events.timestamp = pd.to_datetime(df_events.timestamp)   #convert to date time

    df_events = bucket_events(df_events)

    return df_users, df_events

def bucket_events(df, freq='d'):
    dums = pd.get_dummies(df.event_type)
    new = pd.concat((df, dums), axis=1)
    new = new.set_index(new.timestamp.values)
    new = pd.concat((new.iloc[:,0], new.iloc[:,5:]), axis=1)

    i=0
    for user in new.user.unique():
        # if (i%100==0):
        #      print i
        temp = new[new.user==user]
        bucket_average = pd.DataFrame(np.mean(temp.resample(freq, how='mean')))
        i+=1

    return bucket_average

    
def fit_prelim_model(df):  #for prelim testing purposes
    #read data
    cols = ['public_repos', 'followers','following','public_gists']
    df_small = df[cols]
    X = df_small.values

    rs = ShuffleSplit(X.shape[0], n_iter = 1, random_state=31)

    for train_ind, test_ind in rs:
        train = X[train_ind]
        test = X[test_ind]

    # score_lst = cross_val_score(OneClassSVM(), train, scoring='roc_auc',cv=5)
    # print score_lst
    clf = OneClassSVM()

    #do a parameter grid search
    param_grid = {'kernel': ['rbf'],
                   'nu': [0.3, 0.5, 0.7],
                   'poly': [2, 3],
                   'gamma': [0.1, 0.2, 0.5, 1.0],
                   'coef': [0.05, 0.1] 
                 }

    gs_cv = GridSearchCV(clf, param_grid, n_jobs=-1, scoring='precision').fit(train)
    print gs_cv.best_params_

    #scale data
    scaler = StandardScaler()
    train = scaler.fit_transform(train)

    #clf.fit(train)
    #print clf.predict(test)     #this is pretty heinous
    return None

if __name__ == '__main__':
    #connect_to_mongoDB()
    #df_experts = read_data('data/experts.csv')
    #features = build_features(df_experts)
    df_user, df_events = load_data()
    fit_prelim_model(df_user)