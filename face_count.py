import cv2    #opencv 3.4.2

# Specify the image path for face detection and XML file for the cascade  
 
photo_path = r"C:\Users\panur\.spyder-py3\download.jpg"  #image path, change it based on your imge path
cascade_path = r"C:\Users\panur\.spyder-py3\haarcascade_frontalface_default.xml"  #haarcascade file path, download it and use its path  

# Initialise the Haar Cascade Classifier with the XML file  
haar_face_cascade = cv2.CascadeClassifier(cascade_path)  
  
# Read the photo and convert to grayscale  
photo = cv2.imread(photo_path)  
grayscale = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)  
  
# Detect faces in the photo using OpenCV library  
faces = haar_face_cascade.detectMultiScale(  
    grayscale,  
    scaleFactor = 1.1,  
    minNeighbors = 5,  
    minSize = (30, 30)  
    )  
  
print("Found {0} faces!".format(len(faces)))  
  
# Draw a rectangle around the faces  
for (x, y, w, h) in faces:  
  cv2.rectangle(photo, (x, y), (x+w, y+h), color = (0, 255, 0), thickness = 2)  
  
cv2.imshow("Faces found", photo)  

k=cv2.waitKey(0) # press esc to exit the code
if k==27:
    cv2.destroyAllWindows()