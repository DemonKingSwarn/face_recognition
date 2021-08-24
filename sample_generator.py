import cv2

cam = cv2.VideoCapture(0)
cam.set(1, 600)
cam.set(4, 480)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input("Enter a numeric user ID here: ")

print("Taking samples. Look at camera.......")
count = 0

while True:
    ret, img = cam.read()
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
        count += 1

        cv2.imwrite("samples/faces." + str(face_id) + '.' + str(count) + ".jpg", converted_image[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27: # Press ESC to stop it
        break
    elif count >= 10: # Take 50 samples. More samples -------> More accuracy
        break 

print("Samples taken, now closing the program.")
cam.release()
cv2.destroyAllWindows()