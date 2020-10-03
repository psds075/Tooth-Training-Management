import json
from PIL import Image

BASE_DIR = '../../DB'


# LABEL BOX 파일에서 DB Upload
import os
import json
import cv2
import pymongo

with open('label_numbering.json', encoding='UTF8') as json_file:
    TD_images = json.load(json_file)

# Connection
myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
imageDB = TOOTH_DETECTION["imageDB"]

# 초기화
imageDB.delete_many({})

for image in TD_images[:]:
    FILENAME = image['External ID']


# IMAGE DB Read Query
query = {}
for image in imageDB.find(query):
    print(image)



'''
image = 'sample.jpg'

for size in sizes:
    im = Image.open(image)
    im.thumbnail((100,50))
    im.save("thumbnail_%s_%s" % (str(size), image))
'''