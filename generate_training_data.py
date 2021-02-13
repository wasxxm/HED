import cv2
import imutils as imutils
import numpy as np
import helper
from configs import *
import random


def create_annotated_img(bg_img_file, rect_img_file, bg_img_counter, rect_img_counter):
    bg_img = cv2.imread(bg_img_file, 1)
    img_counter = 1
    # bg_img = helper.add_alpha(bg_img)
    # resize the bg image
    # bg_img_resized = helper.resize_img(bg_img, IMAGE_WIDTH, IMAGE_HEIGHT)
    # bg_img_resized = bg_img
    # helper.display_org("Background", bg_img)
    f = open("annotated.txt", "w")

    rect_img = cv2.imread(rect_img_file, 1)
    rect_img = helper.add_alpha(rect_img)

    if rect_img is None:
        return None

    sizes = [MAX_RECT_WIDTH, random.randint(180, 240), random.randint(130, 170),
             random.randint(80, 120)]

    rect_img_resized = helper.resize_img(rect_img, MAX_RECT_WIDTH)
    h, w, c = rect_img_resized.shape
    if h > w:
        bg_img_resized = helper.resize_img(bg_img, None, h + 10)
    else:
        bg_img_resized = helper.resize_img(bg_img, w + 10, None)

    h, w, c = bg_img_resized.shape

    for size in sizes:

        # print("Size: " + str(size))
        blurs = [round(4 * (size / MAX_RECT_WIDTH))]

        for blur in blurs:

            blur_random = random.randint(0, 1)
            if blur_random == 0:
                blur = 0

            rect_img_resized = helper.resize_img(rect_img, size - (2 * blur))
            # helper.display_org("Rect Resized", rect_img_resized)

            rect_img_blurred = helper.blur_img_edges(rect_img_resized, blur)
            # helper.save_img(rect_img_blurred, BASE_PATH + "1.png")
            # helper.display_org("Blurred", rect_img_blurred)

            height, width, channels = rect_img_blurred.shape
            bordered_rect_img = helper.create_blank_img(width - 1 - (2 * blur), height - 1 - (2 * blur))
            # y = int(height - 1) - blur
            # x = int(width - 1) - blur
            # bordered_rect_img = bordered_rect_img[0:y, 0:x]
            bordered_rect_img = helper.add_border_to_img(bordered_rect_img, 1)
            #
            # helper.display_org("Bordered Rect", helper.add_border_to_img(bordered_rect_img))

            distortions_upper_x = [0, random.randint(1, round(35 * (size / MAX_RECT_WIDTH))),
                                   random.randint(round(40 * (size / MAX_RECT_WIDTH)),
                                                  round(80 * (size / MAX_RECT_WIDTH)))]
            distortions_upper_y = [0, random.randint(1, round(35 * (size / MAX_RECT_WIDTH))),
                                   random.randint(round(40 * (size / MAX_RECT_WIDTH)),
                                                  round(80 * (size / MAX_RECT_WIDTH)))]

            for distortion_index, distortion_upper_x in enumerate(distortions_upper_x):
                distortion_upper_y = distortions_upper_y[distortion_index]
                # print("Distortion Upper X: " + str(distortion_upper_x))
                # print("Distortion Upper y: " + str(distortion_upper_y))
                rect_img_distorted = helper.img_change_perspective2(rect_img_blurred,
                                                                    distort_upper_left=distortion_upper_x,
                                                                    distort_upper_right=distortion_upper_y)

                bordered_distorted_img = helper.img_change_perspective2(bordered_rect_img,
                                                                        distort_upper_left=distortion_upper_x,
                                                                        distort_upper_right=distortion_upper_y)
                # helper.save_img(rect_img_distorted, BASE_PATH + "2.png")

                # helper.display_org("Distorted", rect_img_distorted)
                # helper.display_org("Distorted Bordered", bordered_distorted_img)

                rotations = [0, random.randint(30, 310)]

                for rotation in rotations:

                    # print("Rotation: " + str(rotation))
                    rect_img_rotated = imutils.rotate_bound(rect_img_distorted, rotation)
                    bordered_rotated_img = imutils.rotate_bound(bordered_distorted_img, rotation)
                    # helper.display_org("Rotated", rect_img_rotated)
                    # helper.display_org("Rotated Bordered", bordered_rotated_img)

                    h2, w2, c2 = rect_img_rotated.shape

                    translations_x = [random.randint(abs(int((w - w2) * 0.30)), abs(int((w - w2) * 0.70)))]
                    translations_y = [random.randint(abs(int((h - h2) * 0.30)), abs(int((h - h2) * 0.70)))]

                    # if w2 / w < 0.9:
                    #     translations_x.append(abs(int((w - w2) * 1)))
                    #
                    # if h2 / h < 0.9:
                    #     translations_y.append(abs(int((h - h2) * 1)))

                    for translation_x in translations_x:
                        # print("Translation X: " + str(translation_x))
                        for translation_y in translations_y:

                            # print("Translation Y: " + str(translation_y))
                            intersection = helper.percentage_intersection(bg_img_resized, rect_img_rotated,
                                                                          translation_x,
                                                                          translation_y)
                            # print("Intersection: " + str(intersection))
                            if intersection > 0.80:
                                h, w, c = bg_img_resized.shape
                                blank_bg_img = helper.create_blank_img(w, h)
                                merged_img = helper.overlay_transparent(bg_img_resized, rect_img_rotated,
                                                                        translation_x,
                                                                        translation_y)

                                bg_img_resized_blank = helper.create_blank_img(w, h)
                                merged_img_over_black = helper.overlay_transparent(bg_img_resized_blank, rect_img_rotated,
                                                                                   translation_x,
                                                                                   translation_y)

                                img_annotation = helper.overlay_transparent(blank_bg_img, bordered_rotated_img,
                                                                            translation_x,
                                                                            translation_y)

                                merged_img = helper.resize_img(merged_img, IMAGE_WIDTH, IMAGE_HEIGHT)
                                img_annotation = helper.resize_img(img_annotation, IMAGE_WIDTH, IMAGE_HEIGHT)
                                merged_img_over_black = helper.resize_img(merged_img_over_black, IMAGE_WIDTH, IMAGE_HEIGHT)

                                # helper.display_org("Merged", merged_img)
                                # helper.display_org("Annotated", img_annotation)
                                file_base = BASE_PATH + "generated_v2/" + str(bg_img_counter) + "_" + str(
                                    rect_img_counter) + "_" + str(img_counter)

                                # # ret, thresh = cv2.threshold(imgray, 5, 255, 255)
                                # contours, hierarchy = cv2.findContours(img_annotation, cv2.RETR_EXTERNAL,
                                #                                        cv2.CHAIN_APPROX_SIMPLE)
                                # cnt = contours[0]
                                # img_annotation = cv2.drawContours(blank_bg_img, [cnt], 0, (255, 255, 255), 1)

                                gray = cv2.cvtColor(merged_img_over_black, cv2.COLOR_BGR2GRAY)
                                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                                helper.display_org("Thresh", thresh)
                                edged = cv2.Canny(thresh, 50, 50)

                                # helper.display_org("Edged", threshGrayImage)
                                cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                cnts = sorted(cnts[0], key=cv2.contourArea, reverse=True)[:2]
                                for c in cnts:
                                    ### Approximating the contour
                                    # Calculates a contour perimeter or a curve length
                                    peri = cv2.arcLength(c, True)
                                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                                    # if our approximated contour has four points, then we
                                    # can assume that we have found our screen
                                    screenCnt = approx
                                    if len(approx) == 4:
                                        screenCnt = approx
                                        break
                                    # show the contour (outline)
                                h, w, c = img_annotation.shape
                                blank_bg_img = helper.create_blank_img(w, h)
                                cv2.drawContours(blank_bg_img, [screenCnt], 0, (255, 255, 255), 1)

                                helper.save_img(merged_img, file_base + "_color.jpg")
                                helper.save_img(blank_bg_img, file_base + "_annotation.png")
                                img_counter = img_counter + 1
                                print("Image generated: " + file_base + "_color.jpg")
                                f.write(file_base + "_color.jpg" + "," + file_base + "_annotation.png\n")
                                # helper.display_org("Bordered", bordered_rect_img)
    f.close()
    return True


bg_files = helper.iterate_files(BG_IMG_PATH)
rect_files = helper.iterate_files(RECT_IMG_PATH)

for bg_counter, bg_file in enumerate(bg_files):
    for rect_counter, rect_file in enumerate(rect_files):
        if (create_annotated_img(bg_file, rect_file, bg_counter + 1, rect_counter + 1)) is None:
            continue
        break
    break
