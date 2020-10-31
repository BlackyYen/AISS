# 引用必要套件
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

filePath = 'firebase/detection_rslt.txt'

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('serviceAccount.json')

# 初始化firebase，注意不能重複初始化
firebase_admin.initialize_app(cred)

# 初始化firestore
db = firestore.client()

file = open(filePath,mode='r')

#將txt逐行存入test中
text = []
for line in file:
    text.append(line)    

file.close()

#將標籤和數值存入doc中
doc={}
dict = {}
for data in text:
    data = data.split(',')
    if data[0] in doc:
        doc[data[0]].setdefault(data[1],data[2:])
    else:
        doc[data[0]]={data[1]:data[2:]}
    dict.update(doc)

# 語法
# collection_ref = db.collection("集合路徑")
collection_ref = db.collection("sperm")

# collection_ref提供一個add的方法，input必須是文件，型別是dictionary
collection_ref.add(dict)