# License_plate_detection
this is a moroccan license plate detection and reader
Detection : custom dataset  using labelImg , data preparation and augmentation 
Reader :dataset : https://cc.um6p.ma/cc_datasets
        output : 26178WAW --> 2617و8
         image classification for 16 classes using Yolov8

In general , i wrote a script that from a video that take frames of unique license plates and crop the license plate to be saved in directory crop_img, from input images we go thrpugh all the images in the folder and apply the model to save results in crop_img
Next , for the second model , we take as inputs the crop_img folder to apply charcater segemntation for each plate and detect if it is one the 16 classes.
As outputs , we plot the result images in a grid using matplotlib.pyplot , print the result in the console using character mapping to transform arabic characters back to its original form. Furthermore, the results images are also saved in results Directory . 
