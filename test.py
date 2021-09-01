import cv2
import time
import os
from random import randint
import serial

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app,flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
from firebase_admin import storage

weights='./checkpoints/custom-416.tflite'
size=416
images='./data/images/'
iou=0.45
score=0.50
dont_show=False
info=True
crop_image=False
weed=False
grid_x=0
grid_y=0
gno=""

flags.DEFINE_string('weights', './checkpoints/custom-416.tflite',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_list('images', './data/images/', 'path to input image')
flags.DEFINE_string('output', '', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output') #removable
flags.DEFINE_boolean('info', True, 'print info on detections')
flags.DEFINE_boolean('crop_image', False, 'crop detections from images')
flags.DEFINE_boolean('weed', False, 'perform weed recognition')


cred = credentials.Certificate('./farmbot-1cb00-firebase-adminsdk-k4u4o-9f5057f784.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'farmbot-1cb00.appspot.com'})
db = firestore.client()

def capture():
    print("came to capture")
    cam = cv2.VideoCapture(0)
    img_counter = 0
    while (img_counter<1):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        #cv2.imshow("test", frame)
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(os.path.join("./data/images",img_name), frame)

        #cv2.imwrite('capture' + str(count) + '.png', frame)
        img=cv2.imread("/home/pi/Desktop/tensorflow-yolov4-tflite/data/images/opencv_frame_0.png")
        img_counter += 1
        if(img_counter<1):
            time.sleep(10)
        else:
            break
    cam.release()


def run_detection(gno):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = size
    image_path = "./data/images"
    # tflite load
    interpreter = tf.lite.Interpreter(model_path=weights)

    # loop through images in list and run Yolov4 model on each
    for counter in range(1):
        image_path=os.path.join(image_path,'opencv_frame_'+str(counter)+'.png')
        #image_path=os.path.join(image_path,'weed'+'.png')
        print(image_path)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        # get image name by using split method
        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        #tflite interpret
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # if crop flag is enabled, crop each detection and save it as new image
        if crop_image:
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)


        # if count flag is enabled, perform counting of objects
        counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
        # loop through dict and print
        for key, value in counted_classes.items():
            print("Number of {}s: {}".format(key, value))
        image,x,y = utils.draw_bbox(original_image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes, read_plate =weed)
        print(x)
        print(y)
        image = Image.fromarray(image.astype(np.uint8))
        if not dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite('detect.png', image)
        #print("x:",x,"\ny:",y,"\nvalue:",value,"\n")
        #print("\n Done \n")
        upload(x,y,value,gno)
    return


#function to upload the data to the database
def upload(x_center,y_center,value,gno):

    print("Uploading\n")

    current_time = datetime.now().strftime("%H:%M:%S")
    fileName_local = "detect.png"
    num=randint(10**(10-1), (10**10)-1)
    fileName_server = str(gno)+"_"+str(num)+"_"+fileName_local

    bucket = firebase_admin.storage.bucket()
    if(str(gno)[0:2] == "01"):
        doc_ref = db.collection("farmbed_01").document(str(gno))
    else:
        doc_ref = db.collection("farmbed_02").document(str(gno))
    doc = doc_ref.get()

    if doc.exists:
        dict =doc.to_dict()
        old_file_name=dict['fileName_server']
        #print('Document data:'+ str(dict))
        print("already file exist and file name : ",old_file_name)
        blob_old = bucket.blob(str(old_file_name))
        #print(blob_old)
        blob_old.delete()
        doc.reference.delete()
    #exit()
    else:
        print(u'No such document!')

    blob_new  = bucket.blob(fileName_server)
    blob_new.upload_from_filename(fileName_local)
    blob_new.make_public()
    print(blob_new.public_url)

    if(str(gno)[0:2] == "01"):
        db.collection("farmbed_01").document(str(gno)).set({
        'link':str(blob_new.public_url),
        'fileName_server':fileName_server
        })
    else:
        db.collection("farmbed_02").document(str(gno)).set({
        'link':str(blob_new.public_url),
        'fileName_server':fileName_server
        })
    


    if value==0:
        print("successfully uploaded only to the farmbed")
    else:
        for i in range(value):
            x=x_center[i]
            y=y_center[i]
            type="weeds"
            doc_ref = db.collection(type).document()
            doc_ref.set({
                u'gno':str(gno),
                u'x': str(x),
                u'y': str(y),
                u'time': str(current_time),
                u'url' : str(blob_new.public_url)
            })
            print("successfully uploaded into the farmbed and ",type)
    return

def detect():

    #looping through all the grids
    for i in range (1,3):
        for j in range (1,4):

            #getting the grid number with proper length
            if(len(str(i))==1):
                if(len(str(j))==1):
                    gno="0"+str(i)+"0"+str(j)
                else:
                    gno="0"+str(i)+str(j)
            else:
                if(len(str(j))==1):
                    gno=str(i)+"0"+str(j)
                else:
                    gno=str(i)+str(j)
            print("Grid no : "+gno)

            #moving to the proper grid
            command="MV00X"+gno[0:2]+"Y"+gno[2:4]+"Z00Q00"
            ser.write(bytes(command,"utf-8"))
            print("Pushed the command :",command)
            while(1):
                line = ser.readline().decode('utf-8').rstrip()
                print(line)
                if(line =="ARD_DONE"):
                    break

            #capturing the image from webcam
            capture()

            #running the image processing on the captured image
            run_detection(gno)



if __name__=="__main__":

    #establishing the connection bw arduino and pi
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.flush()

    #running the detect function
    detect()


