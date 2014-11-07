import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pymongo
import subprocess
import requests
import json


def connect_to_mongoDB():
    #set up SSH tunnel to remote server
    subprocess.Popen('ssh -L 27017:dutihr.st.ewi.tudelft.nl:27017 ghtorrent@dutihr.st.ewi.tudelft.nl', 
                     shell=True, close_fds=True)

    #connect to remote mongodb
    client = pymongo.MongoClient("localhost", 27017)
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
    client = MongoClient('mongdodb://localhost:27017/')
    db = client['github_super']

    top_starred = pd.read_csv('../data/top_starred.csv')
    top_projects = top_starred.repository_name.unique()[:150]

    for project in top_projects:
        user = top_projects[repository_name==project].repository_owner.unique()[0]
        url = 'https://api.github.com/repos/' + user + '/' + project + '/stats/contributors'
        r = requests.get(url, auth=('wvanamstel',pwd))

        db.gh.insert()
    pass

def get_github_api_data():
    pass

if __name__ == '__main__':
    #connect_to_mongoDB()
    df_experts = read_data('data/experts.csv')
    features = build_features(df_experts)
    