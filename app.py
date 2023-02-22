import cv2
import os
from flask import Flask,request,render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from joblib import dump, load

app = Flask(__name__)

#Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('model/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

#If these directories don't exist, create folders
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#Total users registered        
def totalreg():
    return len(os.listdir('static/faces'))

#Extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def smile_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smile_points = smile_detector.detectMultiScale(gray, 1.7, 35)
    return smile_points

#Detect face using ML model
def identify_face(facearray):
    model = load('model/face_recognition_model.pkl')
    return model.predict(facearray)

#Trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    dump(knn,'model/face_recognition_model.pkl')
    
    

#Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')
            
def draw_smile(img, center, radius):
    axes = (radius, radius//2)
    color = (255,0,0)
    cv2.ellipse(img, center, axes, 0, 0, 180, color,thickness=2,lineType=cv2.LINE_AA)
    return


################## ROUTING FUNCTIONS #########################

@app.route('/')
def index():
    return render_template('WelcomePage.html')

@app.route('/check')
def home():   
    names,rolls,times,l = extract_attendance()    
    return render_template('Home.html',names=names,rolls=rolls,times=times,l=l,totalreg = totalreg(),datetoday2=datetoday2) 

@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('model'):
        return render_template('Home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    cap = cv2.VideoCapture(0)
    ret = True
    count = 0
    
    while ret:
        ret,frame = cap.read()
        
        face = extract_faces(frame)
        smile = smile_detect(frame)
        
        for (x_face, y_face, w_face, h_face) in face:
            

            
            cv2.rectangle(frame,(x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 20), 2)
            face = cv2.resize(frame[y_face:y_face+h_face,x_face:x_face+w_face], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            
            
            ri_color = frame[y_face:y_face+h_face, x_face:x_face+w_face]
            smiled = False
            
            for (x_smile, y_smile, w_smile, h_smile) in smile:
                draw_smile(ri_color, (x_smile + w_smile // 2, y_smile + h_smile // 4), radius=w_smile // 3)
                smiled =  True

            if smiled:
                cv2.putText(frame,"Smile!",(30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                count += 1
                
                        
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 20, 20),2,cv2.LINE_AA)
            cv2.putText(frame,f'Frame Smiling: {count}/200',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 100, 100),2,cv2.LINE_AA)
            
            userid = identified_person.split('_')[1]
            
            if int(userid) in list(df['Roll']):
                cv2.putText(frame,f'Attendance Added',(30,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.waitKey(2000)
                break

        cv2.imshow('Attendance',frame)
        
        # smile_lst = smile_detector.detectMultiScale(frame, 1.7, 35)
        
        if cv2.waitKey(1)==27:
            break
        
        if count == 200:
            add_attendance(identified_person)
            break


    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()  
    
    return redirect(url_for('home'))

@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x_face,y_face,w_face,h_face) in faces:
            cv2.rectangle(frame,(x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y_face:y_face+h_face,x_face:x_face+w_face])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('Home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
    app.static_folder = 'static'