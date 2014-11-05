import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pymongo
import subprocess

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
    temp=pd.DataFrame()
    for usr in users:
        temp['timestamp'] = df[df.a_actor_attributes_login==usr].a_created_at
        temp['merged_request'] = df[df.a_actor_attributes_login==usr].a_payload_pull_request_merged
        temp.sort('timestamp', inplace=True)
        #temp['merged_request'] = temp['merged_request'].cumsum()

        #construct a time series object
        ts = pd.Series(temp.merged_request.values, index=temp.timestamp)
        ts_res = ts.resample('W', how='sum')
        ts_res.plot()

    return None

def make_experts(df):
    users = df.a_actor_attributes_login.unique()
    metrics={}
    for usr in users:
        temp = df[df.a_actor_attributes_login==usr]
        temp.sort('a_created_at', inplace=True)
        temp = temp.set_index('a_created_at')
        weekly_pulls = np.mean(temp['a_payload_pull_request_merged'].resample('W', how='sum'))
        weekly_commits = np.mean(temp['a_payload_commit_flag'].resample('W', how='sum'))

        metrics['usr']['weekly_pulls'] = weekly_pulls
        metrics['usr']['weekly_commits'] = weekly_commits

    return metrics



if __name__ == '__main__':
    #connect_to_mongoDB()
    df_experts = read_data('data/experts.csv')
    