import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pymongo

def connect_to_mongoDB():
    client = pymongo.MongoClient("localhost", 27017)
    db = client.github
    collection = db.issues

    j=0
    for i in collection.find():
        print i
        j+=1
        if j==5:
            break

if __name__ == '__main__':
    connect_to_mongoDB()