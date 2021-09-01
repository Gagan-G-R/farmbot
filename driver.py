#importing all the reqirments
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
from firebase_admin import storage
from random import randint
import time
import os
import cv2
import serial

#establishing communication bw arduno and pi
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()
  

#getting all the creadentials and refercence to data in firestore
cred = credentials.Certificate('./farmbot-1cb00-firebase-adminsdk-k4u4o-9f5057f784.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'farmbot-1cb00.appspot.com'})
db = firestore.client()
doc_ref = db.collection(u'Status').document(u'Detect')
home_ref = db.collection(u'Status').document(u'Home')


#local function
#this is done if the restart button is clicked in the website ,this is going to del all the data in the database and repopulate with new data collected now
def detect():
    
    print("doing farmbed")
    #running the detect() function in test.py
    os.system('python3 ./test.py')
    home_ref.update({u'state': False})  
    print("farmbed done")

#this is done when detect is not in progress, it does all the command in cmd in the order of time 
def cmd(order,going_home):
    
    print("doing commands")
    #sending all the commands one by one to the arduno 
    for i in order:
        ser.write(bytes(i,"utf-8"))
        print("Pushed command : ",i)
        while(1):
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            if(line =="ARD_DONE"):
                break

    if(going_home):
        home_ref.update({u'state': True})
    else:
        home_ref.update({u'state': False})
    print("commands done")



#this is done if detect is not in progress and there are no commands in cmd also, here the pi is idel
def home():
    print("already @ home")
    #print("already in home")


while(True):

    doc = doc_ref.get()
    home_doc = home_ref.get()
    home_doc = home_doc.to_dict()
    #print(home_doc['state'])

    if doc.exists:

        dict =doc.to_dict()
        if(dict['state']):
            ref = db.collection(u'Status').document(u'Detect')
            ref.update({u'state': False})
            print("turn On : True (pressed detect)")
            detect()   

        else:

            print("turn Off : False (not detecting)")
            cities_ref = db.collection("cmds")
            query = cities_ref.order_by("time")
            results = query.get()
            order=[]

            if(len(results)==0 ):
                if(home_doc['state']):
                    print("came to home the previous command")
                    home()  

                else:
                    print("not @ home")
                    order.append("MV00X00Y00Z00Q00")
                    print(order)
                    cmd(order,True)

            else:
                for doc in results:
                    # print(f'{doc.id} => {doc.to_dict()}')
                    order.append(doc.to_dict()['cmd1'])
                    order.append(doc.to_dict()['cmd2'])
                    print(order)
                    cmd(order,False)

    else:
        print(u'No such document!')
