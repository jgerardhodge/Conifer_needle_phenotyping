#Begin by loading necessary libraries
import os
import csv

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15,15)

import numpy as np
import cv2

###########################################
#ADJUST PIXEL x CM SCALING FACTOR HERE!!!!#
###########################################
px2cm=115.62         

#########################################################
#ADJUST MINIMUM PIXEL AREA FOR CAPTURING LEAVES HERE!!!!#
#########################################################
area_thresh=800   
                  
#######################################
#MODIFY PATH TO IMAGE DATASET HERE!!!!#
#######################################
path='E:/Dropbox/OSU/Projects/PIED_PONR/Corvallis/BNVI/all_cohorts' 

#Change to appropriate directory space and load image
os.chdir(path)

needle_param=[['sample', 'index', 'surface_area', 'major_axis_cm', 'needle_width_minor_axis_cm', 'cylindroid_area', 'cylindroid_volume']]

for fn in os.listdir():
    if '.tif' in fn:
        img=cv2.imread(fn)
        fn_prefix=fn.split('.')[0]
        print(fn)
        
        lab_img=cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        iter = 1 #Arbitrary iterator value required in PlantCV workflows

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_l,img_a,img_b = cv2.split(lab_img)
        
    #This subsection identifies the needle contours within the binary image and estimates the minimum area bounding 
    #rectangle for each.

        #All information appears to be explained by 'l' color channel, the equivalent to 'value' in HSV
        ret, l_thresh = cv2.threshold(img_l, 200, 255, cv2.THRESH_BINARY)
        
        #Draw contours and preserve relevant metadata such as hierarchy
        l_thresh_contours, contours, hierarchy = cv2.findContours(l_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        needle_index=[]
        for c in range(0, len(contours)):                            #Within the binary contours defined, extract the needles
            area=cv2.contourArea(contours[c])
            if area>area_thresh and area<(len(img)*len(img[0])*0.9): #Arbitrary 1000 pixel threshold to remove noisy contours identified 
                needle_index.append(c)                               #Store contours of interest for analysis in 'needle_index'

        img_contours = img.copy()                                    #Create a copy of initial image...

        needle_boxes = []

        for n in needle_index: 
            rect = cv2.minAreaRect(contours[n])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            needle_boxes.append(box)                                   #...(and retain bounding box vertices in 'needle_boxes')...
            cv2.drawContours(img_contours,contours[n],-1,(255,0,0),2)  #...while displaying the contours of each needle (red)...
            cv2.drawContours(img_contours,[box],0,(0,0,255),2)         #...and minimum area bounding box (blue).

        fig=plt.figure(figsize=(30, 9), dpi= 120, facecolor='w', edgecolor='k')
        fig=plt.imshow(img_contours)
        fig=plt.title('l channel thresh')
        fig  
        
    #This section uses the bounding rectangles for each needle to capture both the major axis (in this case an average
    #of the two long sides of the rectangle) and the minor axis which is determined by the mid points of these two sides
    #which are used to count the number of needle containing pixels which exist between them.  These two parameters can
    #then be used to calculate the estimated cylindroid area. All parameters for each needle are then stored in 
    #'needle_param' and are subsequently exported as a CSV file along with the image of the contours and bounding
    #rectangles as a form of record keeping.  

        
        for i in range(0,len(needle_boxes)):
            #Calculate lengths of the four sides of rect for needle box object 'i'
            side1=np.sqrt(np.square(needle_boxes[i][0][1]-needle_boxes[i][1][1])+np.square(needle_boxes[i][0][0]-needle_boxes[i][1][0]))
            side2=np.sqrt(np.square(needle_boxes[i][1][1]-needle_boxes[i][2][1])+np.square(needle_boxes[i][1][0]-needle_boxes[i][2][0]))
            side3=np.sqrt(np.square(needle_boxes[i][2][1]-needle_boxes[i][3][1])+np.square(needle_boxes[i][2][0]-needle_boxes[i][3][0]))
            side4=np.sqrt(np.square(needle_boxes[i][3][1]-needle_boxes[i][0][1])+np.square(needle_boxes[i][3][0]-needle_boxes[i][0][0]))

            #Estimate the midpoints four sides of rect for needle box object 'i'
            midpoint1=[round((needle_boxes[i][0][0]+needle_boxes[i][1][0])/2), round((needle_boxes[i][0][1]+needle_boxes[i][1][1])/2)]
            midpoint2=[round((needle_boxes[i][1][0]+needle_boxes[i][2][0])/2), round((needle_boxes[i][1][1]+needle_boxes[i][2][1])/2)]
            midpoint3=[round((needle_boxes[i][2][0]+needle_boxes[i][3][0])/2), round((needle_boxes[i][2][1]+needle_boxes[i][3][1])/2)]
            midpoint4=[round((needle_boxes[i][3][0]+needle_boxes[i][0][0])/2), round((needle_boxes[i][3][1]+needle_boxes[i][0][1])/2)]

            if side1>side2: #Determine if side1/side3 or side2/side4 represent the major axes
                mid1=midpoint1
                mid2=midpoint3
                major_axis=(side1+side3)/2
                minor_axis=(side2+side4)/2
            else:
                mid1=midpoint2
                mid2=midpoint4
                major_axis=(side2+side4)/2
                minor_axis=(side1+side3)/2

            slope=(mid2[1]-mid1[1])/(mid2[0]-mid1[0]) #Calculate slope once the mid points of the major axes sides are determined

            if (abs(slope)<1): #If slope between midpoints on major rect axes are shallow select pixel positions based on run of m
                run=range(int(mid1[0]),int(mid1[0]+(mid2[0]-mid1[0])))
                pixel_pos=[mid1]
                while pixel_pos[-1][0]<(mid2[0]+1):
                  pixel_pos.append([pixel_pos[-1][0]+1, pixel_pos[-1][1]+slope])
                pixel_pos=np.round(pixel_pos) #Round to ensure pixels can be accurately selected 
                pix_ct=0
                for j in range(0, len(pixel_pos)):
                    if l_thresh[int(pixel_pos[j][1])][int(pixel_pos[j][0])] > 0:
                        pix_ct=pix_ct+1   

                surface_area=cv2.contourArea(contours[needle_index[i]])
                cylindroid_area=(2*np.pi*(pix_ct/2)*major_axis)+(2*np.pi*np.square(pix_ct/2))
                cylindroid_volume=(((pix_ct/2)*(pix_ct/2))*np.pi*major_axis)

                params=[fn_prefix, i+1, surface_area/px2cm, major_axis/px2cm, (minor_axis-pix_ct)/px2cm, cylindroid_area/px2cm, cylindroid_volume/px2cm]
            #else:         #If slope between midpoints on major rect axes are steep select pixel positions based on rise of m
            needle_param.append(params)
            cv2.putText(img_contours, str(i), (int(mid1[0]-10),int(mid1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


        cv2.imwrite(fn_prefix+'_annotated_conts_bound_rect.png', img_contours)

        with open("needle_parameter_list.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(needle_param)