"""import numpy as np"""
import cv2 
import os
from flask import Flask, render_template, Response, request, send_file, redirect


app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2() 
#creates an instance directory for uploading files (/instance/upload)
os.makedirs(os.path.join(app.instance_path, 'upload'), exist_ok=True)

@app.route('/') 
def index():
    
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


@app.route('/image') 
def image_page():
    return render_template('image.html')

@app.route('/upload', methods=('GET','POST'))
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'upload', 'file1'))
        gen2()
    
    return redirect('/image')





def __del__():
    cam.stop()



def gen():    #function for etecting faces on video and webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    

    cam=cv2.VideoCapture(0)   #Cahnge VodioCapture(0) TO VideoCapture(1) if video is not working
    while(cam.isOpened()):    #Change VideoCapture index to 1 incase if video is not working 

        ret, frame=cam.read()

        if not ret:
            frame= cv2.VideoCapture(0) #Cahnge VodioCapture(0) TO VideoCapture(1) if video is not working
            continue
        if ret:

            image= cv2.resize(frame, (0, 0), None, 1, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:

                image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                
                

    
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
        key = cv2.waitKey(20)
        if key == 27:
            break


def gen2():  #function for detecting faces on images
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    frame = cv2.imread('instance/upload/file1')
        
    image= cv2.resize(frame, (0, 0), None, 1, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
            
            

    
    frame = cv2.imencode('.jpg', image)[1].tobytes()
    return (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

        

@app.route('/video_feed') #sending video feed
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_feed') #sending image feed
def image_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')