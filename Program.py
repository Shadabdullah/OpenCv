import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture(0)

cap.set(4, 480)
cap.set(3, 360)


def stoptime():

    print('time function is called reset the timer and allow them to go')

    return None

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        stoptime()

    # Show the frame
    cv2.imshow("Car Detection",img)

    # cv2.imshow("screen", img)
    #
    # SS = car_cascade.detectMultiScale(gray, 1.3, 4)
    #
    # for (x,y,w,h) in SS:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     print('not working')

    key = cv2.waitKey(30)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
