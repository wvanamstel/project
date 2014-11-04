####Connecting to mongo:
> ps -aux | grep mongo   
> kill -9 MONGOPID (if necessary)   
> ssh -L 27017:dutihr.st.ewi.tudelft.nl:27017 ghtorrent@dutihr.st.ewi.tudelft.nl   
> mongo -u ghtorrentro -p ghtorrentro github


####Identify super users for setting the benchmark
What makes a super user?   

* Merged issues   
* Pull requests    
* Number of commits, commits per day    
* Ratio of commits and merged issues
* Anything else?

####Find a small subset of users and their history on gh
Users must have a sufficiently long history on gh, quantified by a certain metric. Perhaps number of commits combined with date when they signed up.

####See if we can detect a changepoint in users' histories
Investigate a wide variety of classification models

####Expand analysis to larger number of users


