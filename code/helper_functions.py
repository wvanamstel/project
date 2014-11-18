# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:38:25 2014

This file contains helper function that were used to acquire user data
after a list of user names were obtained from the github archive using
Google BigQuery: http://www.githubarchive.org/

@author: w
"""
import pandas as pd
import requests
import time
import csv
from itertools import chain


def store_super_users(pwd):
    '''
    Get the details from the top users on github defined by the top 2
    contributors to the 1000 most starred projects
    IN: string: github pwd
    OUT: csv file of top users written to disk
    '''
    top_starred = pd.read_csv('../data/top_projects.csv')
    top_projects = top_starred.repository_name.unique()[:1000]

    data = []
    columns = ['repository', 'user1', 'user2']
    data.append(columns)
    # get top project contributors
    for i in range(len(top_projects)):
        print i
        project = top_projects[i]
        owner = top_starred[top_starred['repository_name']
                            == project].repository_owner.unique()[0]
        url = 'https://api.github.com/repos/' + owner + '/' + project \
              + '/stats/contributors'
        r = requests.get(url, auth=('wvanamstel', pwd))
        time.sleep(1)   # don't overwhelm the api
        records = len(r.json())

        # require that a repo has at least 5 contributors to be included
        if records > 5:
            temp = []
            temp.append(project)
            for i in xrange(1, 3):
                try:
                    temp.append(r.json()[records - i]['author']['login'])
                except TypeError:
                    continue
            data.append(temp)

    # write super users to csv file
    write_to_csv('top_users.csv', data)

    return None


def get_github_user_data(df, pwd):
    ''''
    access github api to scrape user data
    IN: pandas dataframe: user names, string: github api password
    OUT: csv file of user details written to disk
    '''
    users = df.user1.values
    # second_users = df.user2.values
    # all_users = np.hstack((top_users, second_users))

    # get the column names, which are the keys of the json dict
    rndm = requests.get('https://api.github.com/users/mdo',
                        auth=('wvanamstel', pwd))
    cols = rndm.json().keys()
    data = []
    tmp = [['user']]
    tmp.append(cols)
    tmp = list(chain.from_iterable(tmp))
    data.append(tmp)

    for i in xrange(users.shape[0]):
        if i % 100 == 0:
            print i
        tmp = [[users[i]]]
        url = 'https://api.github.com/users/' + str(users[i])
        user_data = requests.get(url, auth=('wvanamstel', pwd))
        tmp.append(user_data.json().values())
        tmp = list(chain.from_iterable(tmp))
        for j in range(len(tmp)):
            try:
                tmp[j] = str(tmp[j])
            except UnicodeError:
                tmp[j] = ''

        data.append(tmp)

    # write to csv
    write_to_csv('bottom_user_details2.csv', data)

    return None


def get_user_events(df, pwd):
    '''
    Access github api to scrape user event data
    IN: pandas DataFrame: user names, string: github api password
    OUT: user events in batches of 100 users
    '''
    # top_users = df.user1.values
    # second_users = df.user2.values
    all_users = df.values  # np.hstack((top_users, second_users))

    # get data
    data = [['user', 'repo', 'event_type', 'action', 'timestamp', 'public']]
    for i in xrange(all_users.shape[0]):
        if i % 10 == 0:
            print i
        if i % 100 == 0:
            write_to_csv('user_events' + str(i/100) + '.csv', data)
            data = [['user', 'repo', 'event_type', 'action', 'timestamp',
                     'public']]

        # Access the github api
        base_url = 'https://api.github.com/users/' + str(all_users[i]) + \
                   '/events?page='
        for j in xrange(1, 11):
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

    # write to csv
    write_to_csv('user_events_last.csv', data)

    return None


def write_to_csv(filename, data):
    '''
    Write intermediate files to csv
    IN: string: target filename, pandas dataframe: data to be written to csv
    OUT: csv files to disk
    '''
    fout = open(filename, 'wb')
    a = csv.writer(fout)
    a.writerows(data)
    fout.close()

    return None


def stitch_together():
    '''
    consolidate user detail and event files
    IN: None
    OUT: csv files written to disk
    '''
    # combine user details
    # df1 = pd.read_csv('../data/raw/top_user_details.csv')
    # df2 = pd.read_csv('../data/raw/top_user_details2.csv')
    # out = pd.concat([df1, df2], axis=0)
    # out.to_csv('../data/user_details_all.csv')

    # combine event files into 1
    df = pd.read_csv('./user_events1.csv')
    for i in range(2, 31):
        filename = './user_events' + str(i) + '.csv'
        df_temp = pd.read_csv(filename)
        df = pd.concat([df, df_temp], axis=0)

    df_last = pd.read_csv('./user_events_last.csv')
    df = pd.concat([df, df_last])
    df.to_csv('../data/bottom_user_events.csv')

    return None
