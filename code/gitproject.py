import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle


'''
This class will load GitHub user data from csv files, and combine them into one
data frame (df_full). Then, both a clustering and a random forest classifie
can be run. Or the user can experiment with other models. After running the
RF classifier, the fitted model is pickled for future use.

Use of the class:
model = GitHub() ---> loads data and constructs dataframe (will take a minute)
model.fit_random_forest(), or:
model.clustering_approach()
'''


class GitHub(object):
    def __init__(self):
        self.df_top_user, self.df_top_events = self.load_data('../data/top_user_details.csv', '../data/top_user_events.csv')
        self.df_user, self.df_event = self.load_data('../data/user_details.csv', '../data/user_events.csv')
        self.df_full = self.make_df()
        self.user_names = self.df_full.pop('user')
        self.labels = self.df_full.pop('super')
        self.columns = self.df_full.columns.values
        self.rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    def make_df(self):
        '''
        Make a single data frame for model input
        IN: dataframe, user details/events
        OUT: dataframe, combined details and events
        '''
        print 'Constructing data frame'
        cols = ['user', 'public_repos', 'followers', 'following',
                'public_gists']
        # construct data frame of super users
        df_small = self.df_top_user[cols]
        df_super = pd.merge(df_small, self.df_top_events, on='user')
        df_super = df_super.drop_duplicates()

        # construct df containing not super users
        df_small = self.df_user[cols]
        df_no_super = pd.merge(df_small, self.df_event, on='user')
        df_no_super = df_no_super.drop_duplicates()

        # label the user as 'super' or not
        df_super['super'] = 1
        df_no_super['super'] = 0
        df_in = pd.concat((df_super, df_no_super), axis=0)

        df_in.public_repos = df_in.public_repos.astype(int)
        df_in.followers = df_in.followers.astype(int)

        # remove very rare event column
        if 'TeamAddEvent' in df_in.columns:
            df_in.drop('TeamAddEvent', axis=1, inplace=True)

        return df_in

    def load_data(self, fin_users, fin_events):
        '''
        Load and preprocess user details and event data
        IN: string, string: filenames of user details and event files
        OUT: dataframes of cleaned up event and details data
        '''
        print 'Loading data'
        df_users = pd.read_csv(fin_users)
        df_events = pd.read_csv(fin_events)

        # clean up data frames
        df_users = df_users[df_users.public_repos != 'False']
        df_users = df_users[df_users.site_admin != 'Not Found']
        df_users = df_users[df_users.public_repos !=
                            'https://developer.github.com/v3/#rate-limiting']

        # clean up user event data
        df_events.drop('Unnamed: 0', axis=1, inplace=True)
        df_events.drop('public', axis=1, inplace=True)
        # clean up repo column
        temp = df_events['repo'].apply(lambda x: x.rsplit()[-1].rstrip('\'}'))
        df_events.repo = temp.apply(lambda x: x[x.find('/') + 1:])
        # convert to date time
        df_events.timestamp = pd.to_datetime(df_events.timestamp)

        # get daily averages of events
        df_events = self.bucket_events(df_events)

        return df_users, df_events

    def bucket_events(self, df, freq='d'):
        '''
        Calculate average daily event frequencies
        IN: dataframe: user event data, string: time frequency (default daily)
        OUT: dataframe of average daily event frequency per user
        '''
        # make dummy variables from the eventtype column
        dums = pd.get_dummies(df.event_type)
        # cols = dums.columns
        new = pd.concat((df, dums), axis=1)
        new = new.set_index(new.timestamp.values)
        # preserve the user column
        new = pd.concat((new.iloc[:, 0], new.iloc[:, 5:]), axis=1)

        # get the frequency of events per time period (default=daily)
        # compute the average daily event frequency
        bucket_average = pd.DataFrame()  # columns=cols)
        for user in new.user.unique():
            temp = new[new.user == user]
            temp2 = pd.DataFrame(np.mean(temp.resample(freq, how='mean'))).transpose()
            temp2['user'] = user
            bucket_average = pd.concat((bucket_average, temp2), axis=0)

        return bucket_average

    def clustering_approach(self):
        '''
        Cluster user data using various clustering algos
        IN: self.df_full and self.labels
        OUT: results to stdout
        '''
        print 'Fitting clustering model'
        X = self.df_full.values
        y = self.labels

        # scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # KMeans
        km_clf = KMeans(n_clusters=2, n_jobs=6)
        km_clf.fit(X)

        # swap labels as super-users are in cluster 0 (messy!!)
        temp = y.apply(lambda x: 0 if x == 1 else 1)
        print '\nKMeans clustering: '
        self.analyse_preds(temp, km_clf.labels_)

        # Agglomerative clustering
        print '\nAgglomerative clustering approach: '
        ac_clf = AgglomerativeClustering()
        ac_labels = ac_clf.fit_predict(X)
        self.analyse_preds(y, ac_labels)

        return None

    def fit_random_forest(self):
        '''
        Do a random forest classification
        IN: self.df_full and self.labels
        OUT: results of RF classification to stdout
        '''
        print 'Fitting Random Forest'
        X = self.df_full.values
        y = self.labels

        # scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=31)

        self.rf_clf.fit(X_train, y_train)
        train_preds = self.rf_clf.predict(X_train)
        self.rf_clf.score(X_test, y_test)
        print '\n Training set: '
        self.analyse_preds(y_train, train_preds)

        # out of sample:
        print '\nTest set: '
        preds = self.rf_clf.predict(X_test)
        self.analyse_preds(y_test, preds)

        # feature importances
        # number of followers, daily issue comments, pull requests: most signal
        feat_imp = pd.DataFrame(np.vstack((self.columns,
                                self.rf_clf.feature_importances_))).transpose()
        feat_imp = feat_imp.sort(columns=1, axis=0, ascending=False)
        print '\nFeature importance: '
        print feat_imp

        # Pickle the fitted Random Forest model for later use
        print '\nPickling model'
        with open('model.pkl', 'w') as f:
            pickle.dump(self.rf_clf, f)

        return None

    def analyse_preds(self, true, pred):
        '''
        Return an analysis of classification results
        IN: numpy array of true values and predicted values
        OUT: metrics to stdout
        '''
        print confusion_matrix(true, pred)
        print 'precision: ', precision_score(true, pred)
        print 'recall: ', recall_score(true, pred)
        print 'roc_auc: ', roc_auc_score(true, pred)

        return None


if __name__ == '__main__':
    model = GitHub()
    model.clustering_approach()
    model.fit_random_forest()
