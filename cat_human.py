# Program   : Human face and cat face identifier
# Author    : Ahmedur Rahman Shovon
# Date      : 30 January, 2017

import cv2, os

# list all files of the input folder
folder_name = "input_test"
img_ar = [folder_name+"/"+s for s in os.listdir(folder_name)]
output_folder = "output_test"

# font for the text written on image
font = cv2.FONT_HERSHEY_SIMPLEX

# These file contains trained classifiers for detecting human face and cat face
human_cas_file = "human.xml"
cat_cas_file = "cat.xml"

# Loading the classifier of faces of human and cat
human_cas = cv2.CascadeClassifier(human_cas_file)
cat_cas = cv2.CascadeClassifier(cat_cas_file)

# Looping through each image of image list
for img_file in img_ar:
    # splits image's file name only and use it as the output image's name
    img_file_only_name = img_file.split("/")[1]
    # reading the image
    img = cv2.imread(img_file)

    # resizing the image keeping the original aspect ratio
    weight, height, channel = img.shape
    new_weight = 500
    aspect_ratio = (new_weight*1.0) / height
    new_height = int(weight*aspect_ratio)
    resize = (new_weight,new_height)
    img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)

    # converts to gray scale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detection of cat face
    cat_face_ar = cat_cas.detectMultiScale(
        img_gray,
        scaleFactor = 1.15,
        minNeighbors = 4,
        minSize = (50,50)
        )

    # detection of human face
    human_face_ar = human_cas.detectMultiScale(
        img_gray,
        scaleFactor = 1.15,
        minNeighbors = 7,
        minSize = (80,80)
        )

    # marking cat faces
    for(x,y,w,h) in cat_face_ar:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"Cat",(x,y-10),font,0.55,(0,255,0),1)

    # marking human faces
    for(x,y,w,h) in human_face_ar:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,"Human",(x,y-10),font,0.55,(0,0,255),1)

    # saves and shows the detected and marked images
    output_file = output_folder+"/out_"+img_file_only_name
    # Comment the below line if you do not want to show marked image on screen
    cv2.imshow(output_file, img)
    cv2.imwrite(output_file,img)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("The processing of the images has been completed.")
print("Check "+output_folder+" folder to see the result.")
