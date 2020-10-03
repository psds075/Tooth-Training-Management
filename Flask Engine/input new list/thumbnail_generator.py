import json
import os
from PIL import Image
import pymongo

BASE_DIR = '../../DB'
THUMB_DIR = '../../THUMB'

# Connection
myclient = pymongo.MongoClient("mongodb://ai:1111@dentiqub.iptime.org:27017/")
TOOTH_DETECTION = myclient["TOOTH_DETECTION"]
imageDB = TOOTH_DETECTION["imageDB"]

with open('label_numbering.json', encoding='UTF8') as json_file:
    TD_images = json.load(json_file)

for image in TD_images[:]:
    FILENAME = image['External ID']
    FILEPATH = os.path.join(BASE_DIR, 'ORIGINAL', FILENAME)
    
    if(not os.path.isfile(FILEPATH)):
        imageDB.delete_one({'FILENAME':FILENAME})
    else:
        im = Image.open(FILEPATH)
        im.thumbnail((100,50))
        im.save(os.path.join(THUMB_DIR, 'ORIGINAL', FILENAME))
        

