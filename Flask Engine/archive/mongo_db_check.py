# -*- coding: utf-8 -*-

import pymongo
import os
import pandas as pd
import json


# Connection
myclient = pymongo.MongoClient("mongodb://ai:1111@127.0.0.1:27017/")


# Image Data Update
DENTIQUB = myclient["DENTIQUB"]
imagedata = DENTIQUB["imagedata"]
dataset = DENTIQUB["dataset"]

# Image DB Load
with open('env.json') as json_file:
    data = json.load(json_file)

BASE_DIR = data['BASE_DIR']
DB_NAME = ''
DB_DIR = BASE_DIR + DB_NAME + '/'

for DIR in os.listdir(BASE_DIR):
    if(os.path.isdir(BASE_DIR+DIR)):
        for FILENAME in os.listdir(BASE_DIR+DIR):
            if not imagedata.find_one({'FILENAME':FILENAME}):
                imagedata.insert_one({'FILENAME':FILENAME,'DATASET_NAME':FILENAME[0:8], 'REVIEW_CHECK': 'UNREAD','CONFIRM_CHECK':'UNCONFIRM'})
                print(FILENAME, 'was inserted to DB.')
            
        if not dataset.find_one({'NAME':DIR}):
            dataset.insert_one({'NAME':DIR,'STATUS':'INSERTED'})
        if(dataset.find({'NAME':DIR}).count()>1):
            print('Duplication Checked.')
        try:
            dataset_num = len(os.listdir(BASE_DIR+DIR))
            if(imagedata.find({'DATASET_NAME':DIR,'CONFIRM_CHECK':'CONFIRM'}).count()+imagedata.find({'DATASET_NAME':DIR,'CONFIRM_CHECK':'DELETE'}).count() == dataset_num):
                myquery = { "NAME":DIR }
                newvalues = { "$set": { "STATUS": "ARCHIVE" } }
                dataset.update_one(myquery, newvalues)
            else:
                myquery = { "NAME":DIR }
                newvalues = { "$set": { "STATUS": "INSERTED" } }
                dataset.update_one(myquery, newvalues)
        except:
            print('exception occured.')

#dataset.update_many({},{ "$set": { "STATUS": "INSERTED" } })
#imagedata.update_many({},{ "$set": { "CONFIRM_CHECK": "UNCONFIRM" } })




