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
print (mat1)

fx = cam.get_camera_information().calibration_parameters.left_cam.fx
fy = cam.get_camera_information().calibration_parameters.left_cam.fy
cx = cam.get_camera_information().calibration_parameters.left_cam.cx
cy = cam.get_camera_information().calibration_parameters.left_cam.cy
distance_between_cameras = cam.get_camera_information().calibration_parameters.T.item(0)

boundaries = [(0, 75, 160), (15, 255, 255)]

runtime_parameters = zcam.PyRuntimeParameters()
runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD

while (True):
    if (cam.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS):
        
        cam.retrieve_image(mat1, sl.PyVIEW.PyVIEW_LEFT)
        cam.retrieve_image(mat2, sl.PyVIEW.PyVIEW_RIGHT)

        left_img = mat1.get_data()
        right_img = mat2.get_data()

        mat1h = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
        mat2h = cv2.cvtColor(right_img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(mat1h, boundaries[0], boundaries[1])
        mask2 = cv2.inRange(mat2h, boundaries[0], boundaries[1])
 
        obj1 = cv2.bitwise_and(mat1h,mat1h, mask= mask1)
        obj2 = cv2.bitwise_and(mat2h,mat2h, mask= mask2)

        #cv2.imshow('left', obj1)        

        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"): 
            break
    
        im2, contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(left_img, contours, -1, (0,255,0), 3)
        max_cnt = max(contours, key = cv2.contourArea)
        cv2.drawContours(left_img, max_cnt, -1, (0,255,0), 3)
        
        cv2.imshow('asf', left_img)
  
        '''
        disparity = abs(lx-rx)
        # print(disparity)
        
        if disparity != 0 :#and shuttlecock.isIncreasing(disparity) == True:
            depth =  (fx * distance_between_cameras) / (-disparity)
            x_val = ((lx - cx)*depth)/fx
            y_val = ((ly - cy)*depth)/fy
        '''
cv2.destroyAllWindows()
    
        
