import pyzed.camera as zcam
import pyzed.core as core
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.mesh as mesh
import numpy as np
import cv2

init = zcam.PyInitParameters()
# coordinate_system=sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_RIGT_HANDED_Y_DOWN
cam = zcam.PyZEDCamera()
if not cam.is_opened():
    print("Opening ZED Camera...")
status = cam.open(init)
if status != tp.PyERROR_CODE.PySUCCESS:
    print(repr(status))
    exit()

runtime = zcam.PyRuntimeParameters()
mat1 = core.PyMat()
mat2 = core.PyMat()

fx = cam.get_camera_information().calibration_parameters.left_cam.fx
fy = cam.get_camera_information().calibration_parameters.left_cam.fy
cx = cam.get_camera_information().calibration_parameters.left_cam.cx
cy = cam.get_camera_information().calibration_parameters.left_cam.cy
distance_between_cameras = cam.get_camera_information().calibration_parameters.T.item(0)

boundaries_l = [(0, 75, 160), (15, 255, 255)]
boundaries_h = [(160, 55, 130), (180, 255, 255)]

runtime_parameters = zcam.PyRuntimeParameters()
runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD

while (True):
    if (cam.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS):
        
        cam.retrieve_image(mat1, sl.PyVIEW.PyVIEW_LEFT)
        cam.retrieve_image(mat2, sl.PyVIEW.PyVIEW_RIGHT)

        left_img = mat1.get_data()
        right_img = mat2.get_data()
        
        blur_left = cv2.GaussianBlur(left_img, (11, 11), 1)
        blur_right = cv2.GaussianBlur(right_img, (11, 11), 1)

        mat1h = cv2.cvtColor(blur_left, cv2.COLOR_BGR2HSV)
        mat2h = cv2.cvtColor(blur_right, cv2.COLOR_BGR2HSV)

        mask11 = cv2.inRange(mat1h, boundaries_l[0], boundaries_l[1])
        mask21 = cv2.inRange(mat2h, boundaries_l[0], boundaries_l[1])
        mask12 = cv2.inRange(mat1h, boundaries_h[0], boundaries_h[1])
        mask22 = cv2.inRange(mat2h, boundaries_h[0], boundaries_h[1])

        mask1 = mask11 + mask12
        mask2 = mask21 + mask22
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)
 
        obj1 = cv2.bitwise_and(mat1h,mat1h, mask= mask1)
        obj2 = cv2.bitwise_and(mat2h,mat2h, mask= mask2)

        #cv2.imshow('left', obj1)

        try:    
            img2, cnt_left, hierarchy_left = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnt_left) != 0:
        
                max_cnt_left = max(cnt_left, key = cv2.contourArea)
                cv2.drawContours(left_img, max_cnt_left, -1, (0,255,0), 3)

                M_l = cv2.moments(max_cnt_left)
                cX_l = int(M_l["m10"] / M_l["m00"])
                cY_l = int(M_l["m01"] / M_l["m00"])
                cv2.circle(left_img, (cX_l, cY_l), 7, (255, 255, 255), -1)
                cv2.putText(left_img, "( " + str(cX_l) +" , " + str(cY_l) + " )" + " , Left", (cX_l + 20, cY_l + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                (center_x_l,center_y_l),radius_l = cv2.minEnclosingCircle(max_cnt_left)
                center_l = (int(center_x_l),int(center_y_l))
                radius_l = int(radius_l)
                cv2.circle(left_img,center_l,radius_l,(255,255,255),2)
                cv2.imshow('Left', left_img)                
        
            img3, cnt_right, hierarchy_right = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnt_right) != 0:
        
                max_cnt_right = max(cnt_right, key = cv2.contourArea)
                cv2.drawContours(right_img, max_cnt_right, -1, (0,255,0), 3)

                M_r = cv2.moments(max_cnt_right)
                cX_r = int(M_r["m10"] / M_r["m00"])
                cY_r = int(M_r["m01"] / M_r["m00"])
                cv2.circle(right_img, (cX_r, cY_r), 7, (255, 255, 255), -1)
                cv2.putText(right_img, "( " + str(cX_r) +" , " + str(cY_r) + " )" + " , Right", (cX_r + 20, cY_r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                (center_x_r,center_y_r),radius_r = cv2.minEnclosingCircle(max_cnt_right)
                center_r = (int(center_x_r),int(center_y_r))
                radius_r = int(radius_r)
                cv2.circle(right_img,center_r,radius_r,(255,255,255),2)
                cv2.imshow('right', right_img)
            
        except ZeroDivisionError:
            print ('object not in camera field')
            break
        
        disparity = abs(center_x_l - center_x_r)
        # print(disparity)        
        if disparity != 0 :#and shuttlecock.isIncreasing(disparity) == True:
            depth =  (fx * distance_between_cameras) / (-disparity)
            x_val = ((center_x_l - cx)*depth)/fx
            y_val = ((center_y_l - cy)*depth)/fy

            print ('\ndisparity = ', disparity)
            print ('depth = ', depth)
            print ('x-coordinate of 3d image = ', x_val)
            print ('y-coordinate of 3d image = ', y_val)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"): 
            break         
        
cv2.destroyAllWindows()

        

    
        
