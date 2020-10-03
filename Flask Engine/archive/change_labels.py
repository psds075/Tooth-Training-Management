import os
import pandas as pd
import json

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
def changelabel(old_label, new_label):
    count = 0
    for DATASET_NAME in folderlist:
        df = pd.read_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', na_rep='')
        if 'BBOX_LABEL' in df:
            for i in range(len(df)):
                if (not df['BBOX_LABEL'].isnull().iloc[i]):
                    BBOX_LABEL = json.loads(df['BBOX_LABEL'].iloc[i])
                    for j in range(len(BBOX_LABEL)):
                        class_name = str(BBOX_LABEL[j]['label'])
                        if(class_name==old_label):
                            BBOX_LABEL[j]['label'] = new_label
                            count += 1
                    df['BBOX_LABEL'].iloc[i] = json.dumps(BBOX_LABEL)
        df.to_excel(BASE_DIR+DATASET_NAME+'.xls', sheet_name='Sheet1', index = False, na_rep='', float_format=None)
    return count

print('하악관침범 수정 : '+changelabel('하악관침범', '하악관중첩'))
print('정중과잉치 수정 : '+changelabel('정중과잉치', '과잉치'))
print('그냥과잉치 수정 : '+changelabel('그냥과잉치', '과잉치'))
print('방사선혼합상 수정 : '+changelabel('방사선혼합상', '방사선혼합성'))
print('FUSION 수정 : '+changelabel('FUSION', 'Fusion'))
print('방사선혼합상 수정 : '+changelabel('방사선혼합상', '방사선혼합성'))

