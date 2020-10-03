import os
import pandas as pd
import json
import collections
import pymongo

'''
# Connection
myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
DENTIQUB = myclient["DENTIQUB"]
imagedata = DENTIQUB["imagedata"]
dataset = DENTIQUB["dataset"]
'''

''' PANDAS 코드
# 디렉토리 불러오기
with open('env.json') as json_file:
    data = json.load(json_file)
    BASE_DIR = data['BASE_DIR']
    
# 폴더명 확인
folderlist = []
for DB_NAME in os.listdir(BASE_DIR):
    if(os.path.isdir(BASE_DIR+DB_NAME)):
        if os.path.isfile(BASE_DIR+DB_NAME+'.xls'):
            folderlist.append(DB_NAME)

LABEL_LIST = []

CONFIRM_COUNT = 0
PREDICT_COUNT = 0
NO_PREDICT_COUNT = 0
NOT_CHECKED = 0

# 개수 카운팅
for DATASET_NAME in folderlist[:]:
    df = pd.read_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', na_rep='')
    if ('BBOX_LABEL' in df) and ('PREDICTION_CHECK' in df):
        for i in range(len(df)):
            if (df['CONFIRM_CHECK'].iloc[i] == 'CONFIRM'):
                CONFIRM_COUNT+=1
                if (df['PREDICTION_CHECK'].iloc[i] == 'PREDICT'):
                    PREDICT_COUNT+=1
                if (df['PREDICTION_CHECK'].iloc[i] == 'NO_PREDICT'):
                    NO_PREDICT_COUNT+=1
                if (df['PREDICTION_CHECK'].iloc[i] == ''):
                    NOT_CHECKED+=1
'''

'''
CONFIRM_COUNT = 0
PREDICT_COUNT = 0
NO_PREDICT_COUNT = 0
NOT_CHECKED = 0

for image in imagedata.find({'CONFIRM_CHECK':'CONFIRM'}):
    CONFIRM_COUNT+=1
    if 'PREDICTION_CHECK' in image:
        if (image['PREDICTION_CHECK'] == 'PREDICT'):
            PREDICT_COUNT+=1
        if (image['PREDICTION_CHECK'] == 'NO_PREDICT'):
            NO_PREDICT_COUNT+=1
        if (image['PREDICTION_CHECK'] == ''):
            NOT_CHECKED+=1
    

CURRENT_PRECISION = int((PREDICT_COUNT+NO_PREDICT_COUNT)/(CONFIRM_COUNT*100+0.1))
CURRENT_DATA_AMOUNT = CONFIRM_COUNT
CURRENT_DATE = 'NOW'
              
print('CONFIRM_COUNT :',CONFIRM_COUNT)
print('PREDICT_COUNT :',PREDICT_COUNT)
print('NO_PREDICT_COUNT :',NO_PREDICT_COUNT)
print('NOT_CHECKED :',NOT_CHECKED)
'''


