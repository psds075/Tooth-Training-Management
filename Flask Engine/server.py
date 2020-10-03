# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
import os
import json
import pandas as pd
import requests
import collections
import cv2
import base64
import numpy as np
import pymongo
from shapely import geometry
from multiprocessing.pool import ThreadPool
from datetime import datetime, date, timedelta
pool = ThreadPool(processes=2)

app = Flask(__name__)
app.secret_key = b'123'
DEBUG_MODE = True
__VERSION__ = '0.1.0'

with open('env.json') as json_file:
    data = json.load(json_file)

BASE_DIR = data['BASE_DIR']
THUMB_DIR = data['THUMB_DIR']
DB_NAME = ''
DB_DIR = BASE_DIR + DB_NAME + '/'


@app.route("/", methods=['GET', 'POST'])
def index():
    if 'ID' in session:
        if(session['ID'] == '매니저'):
            return redirect(url_for('train'))
        else:
            return redirect(url_for('login'))
    else:
        return redirect(url_for('login'))


# 일반 로그인 관련
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if(request.form['ID']=='매니저' and request.form['PASSWORD'] == '1'):
            session['ID'] = '매니저'
            return redirect(url_for('train'))

    session.permanent = True

    return render_template('login.html')

@app.route('/logout')
def logout():
    session['ID'] = False
    return redirect(url_for('login'))

@app.route("/train", defaults={'LISTNAME' : 'NONE'},methods=['GET', 'POST'])
@app.route("/train/<string:LISTNAME>", methods=['GET', 'POST'])
def train(LISTNAME):

    if not 'ID' in session:
        return redirect(url_for('login'))
    if not session['ID'] == '매니저':
        return redirect(url_for('login'))

    ID = session['ID']

    # Connection
    myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
    TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
    imageDB = TOOTH_DETECTION["imageDB"]
    listDB = TOOTH_DETECTION["listDB"]
    
    with open('label_dict.json',encoding = 'utf-8') as json_file:
        data = json.load(json_file)
        LABEL_DICT = data['LABEL_DICT']

    '''
    response = requests.post('http://127.0.0.1:5001/api')
    training_status = json.loads(response.text)['STATUS']

    if(len(training_status.split(' '))==4):
        training_percent = training_status.split(' ')[2]
    else:
        training_percent = 0
    '''
        
    traininglist = []
    archivelist = []

    for data in listDB.find({'ARCHIVE':True}).sort("NAME",pymongo.DESCENDING):
        archivelist.append({'LISTNAME':data['NAME']})

    for data in listDB.find({'ARCHIVE':False}).sort("NAME",pymongo.DESCENDING):
        traininglist.append({'NAME':data['NAME']})
                
    if LISTNAME == 'NONE':
        imagelist = []
        LIST_ARCHIVE = False

    else:
        # 해당 리스트에 UNCONFIRM이 없으면 CONFIRM 시키기
        query = {'LISTNAME':LISTNAME, 'CONFIRM':False}
        if not imageDB.find_one(query):
            query = {'NAME':LISTNAME}
            newvalues = { "$set": { "ARCHIVE": True } }
            listDB.update_one(query, newvalues)
        else:
            query = {'NAME':LISTNAME}
            newvalues = { "$set": { "ARCHIVE": False } }
            listDB.update_one(query, newvalues)
        
        LIST_ARCHIVE = listDB.find_one({'NAME':LISTNAME})['ARCHIVE']

        imagelist = []
        for image in imageDB.find({'LISTNAME' : LISTNAME}):
            if not 'READ' in image:
                image['READ'] = False
            if not 'CONFIRM' in image:
                image['CONFIRM'] = False
            if not 'HOSPITAL' in image:
                image['HOSPITAL'] = 'UNKNOWN'
            if not 'NAME' in image:
                image['NAME'] = 'UNKNOWN'
            data = {
                    'FILENAME' : image['FILENAME'],
                    'READ': image['READ'],
                    'CONFIRM': image['CONFIRM'],
                    'HOSPITAL' : image['HOSPITAL'],
                    'NAME' : image['NAME']
                    }
            imagelist.append(data)
        
    return render_template('train.html', imagelist = imagelist, traininglist = traininglist, archivelist=archivelist, LISTNAME = LISTNAME, LABEL_DICT = json.dumps(LABEL_DICT, ensure_ascii=False), LIST_ARCHIVE=LIST_ARCHIVE, ID = ID)

@app.route("/status", methods=['GET', 'POST'])
def status():
    if not session.get('ID'):
        return redirect(url_for('login'))
    USER = session['ID']

    myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
    TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
    hospitaldata = TOOTH_DETECTION["hospitaldata"]
    hospitals = []

    today = date.today()
    yesterday = today - timedelta(days=1)
    print(yesterday)

    for hospital in hospitaldata.find({}).sort("NAME",pymongo.ASCENDING):
        hospital['WEEKLYIMAGES'] = 0
        for i in range(7):
            searchday = str(today - timedelta(days=i))
            if(searchday in hospital) : hospital['WEEKLYIMAGES'] += hospital[searchday]
        hospital['DAILYIMAGES'] = 0 if not str(today) in hospital else hospital[str(today)]
        if not "최근접속일" in hospital:
            hospital['최근접속일'] = 'NONE'
        if not "최근전송일" in hospital:
            hospital['최근전송일'] = 'NONE'
        hospitals.append(hospital)

    return render_template('status.html', USER=USER, Title = '데이터 수집 현황')


@app.route("/_JSON", methods=['GET', 'POST'])
def sending_data():

    myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
    TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
    imageDB = TOOTH_DETECTION["imageDB"]
    hospitaldata = TOOTH_DETECTION["hospitaldata"]
    listDB = TOOTH_DETECTION["listDB"]

    if(request.json['ORDER'] == 'GET'):
        target = imageDB.find_one({'FILENAME':request.json['FILENAME']})
        if not 'TMJ_LEFT' in target:
            target['TMJ_LEFT'] = ''
        if not 'TMJ_RIGHT' in target:
            target['TMJ_RIGHT'] = ''
        if not 'OSTEOPOROSIS' in target:
            target['OSTEOPOROSIS'] = ''
        if not 'COMMENT_TEXT' in target:
            target['COMMENT_TEXT'] = ''
        if not 'READ' in target:
            target['READ'] = False
        if not 'BBOX_LABEL' in target:
            target['BBOX_LABEL'] = '[]'
        if not 'CONFIRM' in target:
            target['CONFIRM'] = False
        if not 'PREDICTION_CHECK' in target:
            target['PREDICTION_CHECK'] = 'NO_PREDICT'
        data = {'FILENAME' : target['FILENAME'], 
                'TMJ_LEFT':target['TMJ_LEFT'], 
                'TMJ_RIGHT':target['TMJ_RIGHT'],
                'OSTEOPOROSIS':target['OSTEOPOROSIS'], 
                'COMMENT_TEXT':target['COMMENT_TEXT'],
                'READ':target['READ'],
                'BBOX_LABEL':target['BBOX_LABEL'],
                'CONFIRM':target['CONFIRM'],
                'PREDICTION_CHECK':target['PREDICTION_CHECK']
                } 
        
        target['READ'] = True
        if(request.json['ID'] != 'MANAGER'): #버그픽스
            today = str(date.today())
            hospitaldata.update_one({'NAME':request.json['ID']}, { "$set": {"최근접속일": today} })
        imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": target })

        return json.dumps(json.dumps(data))

    if(request.json['ORDER'] == 'SET'):

        if(request.json['PARAMETER'] == 'BBOX_LABEL'):
            BBOX_LABEL = json.loads(request.json['SETVALUE'])
            for EACH_LABEL in BBOX_LABEL:
                print(EACH_LABEL)
            for EACH_LABEL in BBOX_LABEL:
                EACH_LABEL['left'] = int(EACH_LABEL['left'] / request.json['RATIO'])
                EACH_LABEL['top'] = int(EACH_LABEL['top'] / request.json['RATIO'])
                EACH_LABEL['width'] = int(EACH_LABEL['width'] / request.json['RATIO'])
                EACH_LABEL['height'] = int(EACH_LABEL['height'] / request.json['RATIO'])
            for EACH_LABEL in BBOX_LABEL:
                print(EACH_LABEL)
            imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": {request.json['PARAMETER']:json.dumps(BBOX_LABEL)}})

        elif(request.json['PARAMETER'] == 'CONFIRM'):
            imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": {request.json['PARAMETER']:request.json['SETVALUE']}})
            # Confirm 관련 데이터셋의 경우 시간까지 기록함
            imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": {'TIMESTAMP':str(pd.Timestamp('now'))}})
            # 전체 데이터셋이 Confirm인 경우 Dataset의 Status 바꿈
            if(not imageDB.find_one({'LISTNAME':request.json['DATASET'],'CONFIRM':False})):
                listDB.update_one({'NAME':request.json['DATASET']},{ "$set": { "ARCHIVE": True } })

        else:
            imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": {request.json['PARAMETER']:request.json['SETVALUE']}})

        return json.dumps('Success')

    if(request.json['ORDER'] == 'PREDICTION'):

        myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
        TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
        imageDB = TOOTH_DETECTION["imageDB"]
        listDB = TOOTH_DETECTION["listDB"]

        query = {'LISTNAME':request.json['DATASET'], 'FILENAME':request.json['FILENAME']}
        target_image = imageDB.find_one(query)
        
        if('BBOX_PREDICTION' not in target_image) or (request.json['PARAMETER'] == 'FORCE'):
            imagepath = os.path.join(BASE_DIR+request.json['DATASET'],request.json['FILENAME'])
            img = hanimread(imagepath) #img = cv2.imread(target_image) 대체함. 한글경로 버그 수정
            data = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
            mydata = {'img_name' : request.json['FILENAME'], 'data' : data}

            #병렬 코드
            boxes1 = pool.apply_async(request_prediction, (5101, mydata)) 
            boxes2 = pool.apply_async(request_prediction, (5102, mydata))
            #osteoporosis = pool.apply_async(request_prediction, (5201, mydata))
            boxes = boxes1.get()['BOXES'] + boxes2.get()['BOXES']
            boxes = bbox_duplicate_check(boxes)
            VERSION = boxes1.get()['VERSION']
            ARCHITECTURE = boxes1.get()['ARCHITECTURE']
            TRAINING_DATE = boxes1.get()['TRAINING_DATE']

            if(DEBUG_MODE == True):
                #print(type(boxes), boxes)
                pass 

            imageDB.update_one({'FILENAME':request.json['FILENAME']}, { "$set": {'BBOX_PREDICTION':json.dumps(boxes),'BBOX_VERSION':VERSION, 'BBOX_ARCHITECTURE':ARCHITECTURE,'TRAINING_DATE':TRAINING_DATE}})

        else:
            boxes = json.loads(target_image['BBOX_PREDICTION'])
            
        return json.dumps(boxes)

    if(request.json['ORDER'] == 'START_TRAINING'):
        response = requests.post('http://dentiqub.iptime.org:5001/start')
        if(DEBUG_MODE == True):
            print(json.loads(response.text)['STATUS'])
        return json.dumps(json.loads(response.text)['STATUS'])

    if(request.json['ORDER'] == 'TRAINING_STATUS'):
        response = requests.post('http://dentiqub.iptime.org:5001/api')
        if(DEBUG_MODE == True):
            print(json.loads(response.text)['STATUS'])
        return json.dumps(json.loads(response.text)['STATUS'])

    if(request.json['ORDER'] == 'STATISTICS'):
        LABEL_RANK = label_statistics()
        PRECISION_DATA = prediction_statistics()
        return json.dumps({'LABEL_RANK':LABEL_RANK, 'PRECISION_DATA':PRECISION_DATA})

    if(request.json['ORDER'] == 'STATISTICS_DEMO'):
        LABEL_RANK = label_statistics_demo()
        PRECISION_DATA = prediction_statistics_demo()
        print('Sending Complete.')
        return json.dumps({'LABEL_RANK':LABEL_RANK, 'PRECISION_DATA':PRECISION_DATA})

@app.route('/database/<path:path>')
def database(path):
    return send_from_directory(BASE_DIR, path) 

@app.route('/thumb/<path:path>')
def thumb_database(path):
    return send_from_directory(THUMB_DIR, path)

# Necessary Functions
def hanimread(filePath):
    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def bbox2rect(bbox):
    top = bbox['top']
    down = bbox['top'] + bbox['height']
    left = bbox['left']
    right = bbox['left'] + bbox['width']
    return [[left, top], [left, down], [right, down], [right, top]] 

def calculate_cross(box_1, box_2):
    poly_1 = geometry.Polygon(box_1)
    poly_2 = geometry.Polygon(box_2)
    cross_a = poly_1.intersection(poly_2).area / poly_1.area
    cross_b = poly_1.intersection(poly_2).area / poly_2.area
    cross = cross_a if cross_a > cross_b else cross_b
    return cross

def combine_bbox(bbox_1, bbox_2):
    top_1 = bbox_1['top']
    down_1 = bbox_1['top'] + bbox_1['height']
    left_1 = bbox_1['left']
    right_1 = bbox_1['left'] + bbox_1['width']
    
    top_2 = bbox_2['top']
    down_2 = bbox_2['top'] + bbox_2['height']
    left_2 = bbox_2['left']
    right_2 = bbox_2['left'] + bbox_2['width']
    
    top = top_1 if top_1 < top_2 else top_2
    left = left_1 if left_1 < left_2 else left_2
    down = down_1 if top_1 > top_2 else down_2
    right = right_1 if right_1 > right_2 else right_2
    
    bbox = {'left' : left, 'top' : top, 'width':right-left, 'height':down - top, 'label' : bbox_1['label']}
    
    return bbox
    
def bbox_duplicate_check(boxes):
    while(1):
        if(len(boxes) > 1):
            check = False
            for i in range(len(boxes)):
                if(check):
                    break
                for j in range(i+1, len(boxes)):
                    if(boxes[i]['label'] == boxes[j]['label']):
                        if(calculate_cross(bbox2rect(boxes[i]), bbox2rect(boxes[j])) > 0.3):
                            print(calculate_cross(bbox2rect(boxes[i]), bbox2rect(boxes[j])))
                            boxes.append(combine_bbox(boxes[i], boxes[j]))
                            del boxes[j], boxes[i] 
                            check = True
                            break
        else:
            break
        if(check == False):
            break
    return boxes

def request_prediction(port, mydata):
    response = requests.post('http://dentiqub.iptime.org:'+str(port)+'/api', json=mydata)
    boxes = json.loads(response.text)['message']
    VERSION = json.loads(response.text)['VERSION']
    ARCHITECTURE = json.loads(response.text)['ARCHITECTURE']
    TRAINING_DATE = json.loads(response.text)['TRAINING_DATE']
    return {'BOXES':boxes, 'VERSION':VERSION, 'ARCHITECTURE':ARCHITECTURE, 'TRAINING_DATE':TRAINING_DATE}

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host = '0.0.0.0', port = 81)


