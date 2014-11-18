Repo containing the capstone project for Zipfian Academy's Data Science program.

The object of this project is to identify coding talent on GitHub looking at a user's social activity. This can be useful for recruiting purposes, or to track the progress of a user's coding ability.

Taking the top 1000 most starred projects on GitHub and using the top 2 contributors to these projects, about 1100 so-called 'super-users' were identified representing top coding talent. In addition, 2200 randomly chosen user profiles having less social activity were used to complete the data set.  Code for this is contained in the helper_functions.py file.

User names were obtained via http://githubarchive.org and Google BigQuery, user details were scraped from github using their API.

The main gitproject.py file contains code using various supervised and unsupervised models to investigate the data set printing results to standard out. 

Currently, the model differentiates between the two classes of users with a precision of 0.89 and a recall of 0.83 on the test set (25% of the total data set).

To run from the command line:
> ipython gitproject.py

Next step is to allow an input of a GitHub user name and return how far along this user is in becoming a super user. 