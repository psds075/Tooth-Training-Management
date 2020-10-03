# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for
import os
import json
import pandas as pd
import requests
import collections
import cv2
import base64
import numpy as np

app = Flask(__name__)
DEBUG_MODE = True

with open('env.json') as json_file:
    data = json.load(json_file)

BASE_DIR = data['BASE_DIR']
DB_NAME = ''
DB_DIR = BASE_DIR + DB_NAME + '/'
TABLE_LIST = ['GUIDED_FILENAME','SEX','AGE','STATUS','TMJ_LEFT','TMJ_RIGHT','OSTEOPOROSIS','COMMENT_TEXT','REVIEW_CHECK','BBOX_LABEL', 'CONFIRM_CHECK','PREDICTION_CHECK','TIMESTAMP']

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/viewer", defaults={'DATASET_NAME' : 'NONE'},methods=['GET', 'POST'])
@app.route("/viewer/<string:DATASET_NAME>", methods=['GET', 'POST'])
def viewer(DATASET_NAME):
    with open('label_dict.json',encoding = 'utf-8') as json_file:
        data = json.load(json_file)
        LABEL_DICT = data['LABEL_DICT']

    response = requests.post('http://127.0.0.1:5002/api')
    training_status = json.loads(response.text)['STATUS']
    if(len(training_status.split(' '))==4):
        training_percent = training_status.split(' ')[2]
    else:
        training_percent = 0
    datasetlist = []
    archivelist = []
    for DIR in os.listdir(BASE_DIR):
        if(os.path.isdir(BASE_DIR+DIR)):
            if not os.path.isfile(BASE_DIR+DIR+'.xls'):
                df = pd.DataFrame({'FILENAME':os.listdir(BASE_DIR+DIR)})
                df.to_excel(BASE_DIR+DIR+'.xls', sheet_name='Sheet1', index = False, float_format=None)
            try:
                df = pd.read_excel(BASE_DIR+DIR+'.xls', sheet_name='Sheet1', na_rep='')
                if 'CONFIRM_CHECK' in df:
                    if((df['CONFIRM_CHECK'] == 'CONFIRM').sum() + (df['CONFIRM_CHECK'] == 'DELETE').sum() != len(df)):
                        datasetlist.append({'DATASET_NAME' : DIR})
                    elif(len(os.listdir(BASE_DIR+DIR)) != len(df)):
                        datasetlist.append({'DATASET_NAME' : DIR})
                    else:
                        archivelist.append({'DATASET_NAME' : DIR})
                else:
                    datasetlist.append({'DATASET_NAME' : DIR})
            except:
                print('exception occured.')
    if DATASET_NAME == 'NONE':
        datalist = []

    else:
        global DB_DIR
        global DB_NAME
        DB_NAME = DATASET_NAME
        DB_DIR = BASE_DIR + DATASET_NAME + '/'
        if not os.path.isfile(BASE_DIR+DATASET_NAME+'.xls'):
            df = pd.DataFrame({'FILENAME':os.listdir(BASE_DIR+DATASET_NAME)})
            df.to_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', index = False, float_format=None)
        df = pd.read_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', na_rep='')
        df = df.fillna('')

        check_row = False
        for filename in os.listdir(BASE_DIR+DATASET_NAME):
            if not (filename in list(df.FILENAME)):
                df = df.append({"FILENAME":filename}, ignore_index=True)
                check_row = True
        if(check_row == True):
            df = df.fillna('')
            df = df.sort_values(by=["FILENAME"])
            df.to_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)

        check_column = False
        for column in TABLE_LIST:
            if not column in df.columns.tolist():
                check_column = True
                df[column] = ''
        if(check_column == True):
            df = df.fillna('')
            df.to_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)

        datalist = []
        df = df.sort_values(by=["FILENAME"])
        for i in range(len(df)):
            if(df['REVIEW_CHECK'].iloc[i] == ''):
                df['REVIEW_CHECK'].iloc[i] = 'UNREAD'
            if(df['CONFIRM_CHECK'].iloc[i] == ''):
                df['CONFIRM_CHECK'].iloc[i] = 'UNCONFIRM'
            data = {'FILENAME' : df['FILENAME'].iloc[i],
                    'REVIEW_CHECK':str(df['REVIEW_CHECK'].iloc[i]),
                    'CONFIRM_CHECK':str(df['CONFIRM_CHECK'].iloc[i])
                    }
            datalist.append(data)
        df.to_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)

    return render_template('viewer.html', datalist = datalist, datasetlist = datasetlist, archivelist=archivelist, current_dataset = DATASET_NAME, LABEL_DICT = json.dumps(LABEL_DICT, ensure_ascii=False), training_status = training_status, training_percent = training_percent)

@app.route("/_JSON", methods=['GET', 'POST'])
def sending_data():
    if(request.json['ORDER'] == 'LIST'):
        df = pd.read_excel(BASE_DIR+request.json['DATASET']+'.xls', sheet_name='Sheet1', na_rep='')
        datalist = []
        for i in range(len(df)):
            data = {'FILENAME' : df['FILENAME'].iloc[i], 
                    'REVIEW_CHECK':str(df['REVIEW_CHECK'].iloc[i]),
                    'CONFIRM_CHECK':str(df['CONFIRM_CHECK'].iloc[i]),
                    }
            datalist.append(data)
        return json.dumps(datalist)

    if(request.json['ORDER'] == 'TARGET'):
        df = pd.read_excel(BASE_DIR+request.json['DATASET']+'.xls', sheet_name='Sheet1')
        df = df.fillna('')
        i = df.index[df['FILENAME'] == request.json['FILENAME']].tolist()[0]
        data = {'FILENAME' : df['FILENAME'].iloc[i], 
                'TMJ_LEFT':str(df['TMJ_LEFT'].iloc[i]), 
                'TMJ_RIGHT':str(df['TMJ_RIGHT'].iloc[i]),
                'OSTEOPOROSIS':str(df['OSTEOPOROSIS'].iloc[i]), 
                'COMMENT_TEXT':str(df['COMMENT_TEXT'].iloc[i]),
                'REVIEW_CHECK':str(df['REVIEW_CHECK'].iloc[i]),
                'BBOX_LABEL':str(df['BBOX_LABEL'].iloc[i]),
                'CONFIRM_CHECK':str(df['CONFIRM_CHECK'].iloc[i]),
                'PREDICTION_CHECK':str(df['PREDICTION_CHECK'].iloc[i])
                }
        df['REVIEW_CHECK'].iloc[i]='READ'
        df.to_excel(BASE_DIR+request.json['DATASET']+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)
        return json.dumps(json.dumps(data))

    if(request.json['ORDER'] == 'LABEL'):
        df = pd.read_excel(BASE_DIR+request.json['DATASET']+'.xls', sheet_name='Sheet1', na_rep='')
        if(DEBUG_MODE == True):
            pass
            #print(df)
        i = df.index[df['FILENAME'] == request.json['FILENAME']].tolist()[0]
        print(request.json['PARAMETER'],str(request.json['SETVALUE']))
        if(DEBUG_MODE == True):
            print(request.json['PARAMETER'],str(request.json['SETVALUE']))
        df[request.json['PARAMETER']].iloc[i]=str(request.json['SETVALUE'])
        if(request.json['PARAMETER'] == 'BBOX_LABEL'):
            if(df[request.json['PARAMETER']].iloc[i] == ''):
                df[request.json['PARAMETER']].iloc[i] = '[]'
            BBOX_LABEL = json.loads(df[request.json['PARAMETER']].iloc[i])
            for EACH_LABEL in BBOX_LABEL:
                EACH_LABEL['left'] = int(EACH_LABEL['left'] / request.json['RATIO'])
                EACH_LABEL['top'] = int(EACH_LABEL['top'] / request.json['RATIO'])
                EACH_LABEL['width'] = int(EACH_LABEL['width'] / request.json['RATIO'])
                EACH_LABEL['height'] = int(EACH_LABEL['height'] / request.json['RATIO'])
            df[request.json['PARAMETER']].iloc[i] = json.dumps(BBOX_LABEL)
        if(request.json['PARAMETER'] == 'CONFIRM_CHECK'):
            df['TIMESTAMP'].iloc[i] = str(pd.Timestamp('now'))
        if(DEBUG_MODE == True):
            pass
            #print(df)
        df.to_excel(BASE_DIR+request.json['DATASET']+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)
        return json.dumps('Success')

    if(request.json['ORDER'] == 'PREDICTION'):
        target_image = os.path.join(BASE_DIR+request.json['DATASET'],request.json['FILENAME'])
        img = hanimread(target_image) #img = cv2.imread(target_image) 대체
        data = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
        mydata = {'img_name' : request.json['FILENAME'], 'data' : data}
        response = requests.post('http://127.0.0.1:5001/api', json=mydata)
        if(DEBUG_MODE == True):
            print(json.loads(response.text)['message'])
        return json.dumps(json.loads(response.text)['message'])

    if(request.json['ORDER'] == 'START_TRAINING'):
        response = requests.post('http://127.0.0.1:5002/start')
        if(DEBUG_MODE == True):
            print(json.loads(response.text)['STATUS'])
        return json.dumps(json.loads(response.text)['STATUS'])

    if(request.json['ORDER'] == 'TRAINING_STATUS'):
        response = requests.post('http://127.0.0.1:5002/api')
        if(DEBUG_MODE == True):
            print(json.loads(response.text)['STATUS'])
        return json.dumps(json.loads(response.text)['STATUS'])


    if(request.json['ORDER'] == 'STATISTICS'):
        LABEL_RANK = label_statistics()
        PRECISION_DATA = prediction_statistics()
        return json.dumps({'LABEL_RANK':LABEL_RANK, 'PRECISION_DATA':PRECISION_DATA})

@app.route('/database/<path:path>')
def database(path):
    return send_from_directory(BASE_DIR, path) 

def label_statistics():
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
    
    return LABEL_RANK

def prediction_statistics():
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

    CONFIRM_COUNT = 0
    PREDICT_COUNT = 0
    NO_PREDICT_COUNT = 0
    NOT_CHECKED = 0

    # PRECISION 카운팅
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

    CURRENT_PRECISION = int((PREDICT_COUNT+NO_PREDICT_COUNT)/CONFIRM_COUNT*100)
    CURRENT_DATA_AMOUNT = CONFIRM_COUNT
    CURRENT_DATE = 'NOW'

    # 디렉토리 불러오기
    with open('precision_record.json') as json_file:
        data = json.load(json_file)
        PRECISION_RECORD = data['PRECISION_RECORD']

    PRECISION_DATA = []
    for PRECISION in PRECISION_RECORD:
        PRECISION_DATA.append((PRECISION[0],PRECISION[1], PRECISION[2]))
    PRECISION_DATA.append((CURRENT_DATE, CURRENT_DATA_AMOUNT, str(CURRENT_PRECISION) + '%'))

    return PRECISION_DATA

def hanimread(filePath):
    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host = '0.0.0.0', port = 80)


