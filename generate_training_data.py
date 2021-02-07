import cv2
import numpy as np
import helper
from configs import *

bg_img = cv2.imread(BG_IMG_PATH + "2.jpg", 1)
# bg_img = helper.add_alpha(bg_img)

helper.display("Bg", bg_img)
# resize the bg image
bg_img_resized = helper.resize_img(bg_img, IMAGE_WIDTH, IMAGE_HEIGHT)
# display("BgRes", bg_img_resized)

rect_img = cv2.imread(RECT_IMG_PATH + "credit_cards05.jpg", 1)
rect_img = helper.add_alpha(rect_img)
# display("Rect", rect_img)
rect_img_resized = helper.resize_img(rect_img, 248)
# display_org("RectRes", rect_img_resized)

output = helper.blur_img_edges(rect_img_resized, BLUR_EDGE_RADIUS)

helper.display_org("Blur", output)
# helper.save_img(output, BASE_PATH + "1.png")

# merged_img = overlay_img(bg_img_resized, output, 15, 15)
merged_img = helper.overlay_transparent(bg_img_resized, output, 15, 15)
helper.display("RectMerged", merged_img)
