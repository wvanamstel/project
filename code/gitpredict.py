import cPickle as pickle
import pandas as pd
import numpy as np
import requests
import time

'''
Work in progress
'''


class GitPredict(object):
    def __init__(self):
        # load pickled random forest model
        self.rf_clf = pickle.load(open('model.pkl', 'rb'))
        self.user_details = []
        self.get_user_details()

    def get_user_details(self):
        print 'Enter GitHub user name: '
        user_name = 'wvanamstel' #raw_input()
        url = 'https://api.github.com/users/' + user_name
        print 'Get data from GitHub'
        
        #acces github api for user details
        temp_details = requests.get(url)
        
        # columns from the user details that are used
        cols = ['public_repos', 'followers', 'following',
                'public_gists']
        
        for col in cols:
            self.user_details.append(temp_details.json()[col])
        
        # get user's events
        url = 'https://api.github.com/users/' + user_name + '/' + 'events?page='
        events = []
        for j in xrange(1, 11):
            url = url + str(j)
            try:
                event_data = requests.get(url)
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
                    events.append(temp)
                    
        df_event = pd.DataFrame(events)
        # make dummy variables from the eventtype column
        dums = pd.get_dummies(df_event.event_type)
        # cols = dums.columns
        new = pd.concat((df_event, dums), axis=1)
        new = new.set_index(new.timestamp.values)
        # preserve the user column
        new = pd.concat((new.iloc[:, 0], new.iloc[:, 5:]), axis=1)

        # get the frequency of events per time period (default=daily)
        # compute the average daily event frequency
        bucket_average = pd.DataFrame()  # columns=cols)
        for user in new.user.unique():
            temp = new[new.user == user]
            temp2 = pd.DataFrame(np.mean(temp.resample('d', how='mean'))).transpose()
            temp2['user'] = user
            bucket_average = pd.concat((bucket_average, temp2), axis=0)
