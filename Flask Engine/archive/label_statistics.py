import os
import pandas as pd
import json
import collections
import pymongo

# Connection
myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
DENTIQUB = myclient["DENTIQUB"]
imagedata = DENTIQUB["imagedata"]
dataset = DENTIQUB["dataset"]


LABEL_LIST = []

for image in imagedata.find({'CONFIRM_CHECK':'CONFIRM'}):
    if ('BBOX_LABEL' in image):
        if((image['BBOX_LABEL']) == ''):
            image['BBOX_LABEL'] = '[]'
        BBOX_LABEL = json.loads(image['BBOX_LABEL'])
        for j in range(len(BBOX_LABEL)):
            class_name = str(BBOX_LABEL[j]['label'])
            LABEL_LIST.append(class_name)

LABEL_COUNTER = collections.Counter(LABEL_LIST)
LABEL_RANK = []

for key, value in sorted(LABEL_COUNTER.items(), key=lambda item: item[1], reverse = True):
    LABEL_RANK.append((key, value))

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

# 개수 카운팅
for DATASET_NAME in folderlist[:]:
    df = pd.read_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', na_rep='')
    if 'BBOX_LABEL' in df:
        for i in range(len(df)):
            if (not df['BBOX_LABEL'].isnull().iloc[i]) and (df['CONFIRM_CHECK'].iloc[i] == 'CONFIRM'):
                BBOX_LABEL = json.loads(df['BBOX_LABEL'].iloc[i])
                for j in range(len(BBOX_LABEL)):
                    class_name = str(BBOX_LABEL[j]['label'])
                    LABEL_LIST.append(class_name)

LABEL_COUNTER = collections.Counter(LABEL_LIST)
LABEL_RANK = []

for key, value in sorted(LABEL_COUNTER.items(), key=lambda item: item[1], reverse = True):
    LABEL_RANK.append((key, value))

'''



