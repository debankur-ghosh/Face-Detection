# Face-Detection
## Steps to run the program
**install dependencies**
This program requires python 3.6 to run
install pip if not already installed

1. pip install flask
2. pip install opencv-python

use pip3 instead of pip if you have pip version 3

**Run the program**
1. git clone https://github.com/debankur-ghosh/Face-Detectionhttps://github.com/debankur-ghosh/Face-Detection.git
2. FLASK_APP=app.py
3. flask run
---
## Follow the bellow steps if you're using anaconda
1. conda create --name flsk python=3.6 -y
2. conda activate flsk
3. pip install flask
4. pip install opencv-python
5. FLASK_APP=app.py
6. flask run


# result 

**index page** 
![alt text](index.png)


**using webcam**
![alt text](webcam.png)


**using photo**
![alt text](image.png)



## Incase webcam page is not working or not showing output
edit line number 48 of app.py as shown below
edit the video capture index to 1

**change this**
```python
cam=cv2.VideoCapture(0)  
    while(cam.isOpened()):    
        ret, frame=cam.read()

        if not ret:
            frame= cv2.VideoCapture(0) 
  ```          
            
            
**to this**
``` python
cam=cv2.VideoCapture(1)   
    while(cam.isOpened()):    
        ret, frame=cam.read()

        if not ret:
            frame= cv2.VideoCapture(1) 
            continue ```


