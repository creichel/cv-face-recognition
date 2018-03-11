#######################################################
# Face UI methods
# @author: Christian Reichel
# Version: 0.2
# -----------------------------------------------------
# Library for common UI things for object recognition
#######################################################

# IMPORTS
import cv2 as cv 	# For drawing functions

def labeled_box(image, upper_label, lower_label, from_pos, to_pos, boundary_box_thickness = 2):
    cv.putText(image, upper_label, (from_pos[0], from_pos[1] - boundary_box_thickness * 2), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(image, lower_label, (from_pos[0], to_pos[1] + boundary_box_thickness * 7), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv.LINE_AA)
    image = cv.rectangle(image, from_pos, to_pos, (255, 0, 0), boundary_box_thickness)

    return image
