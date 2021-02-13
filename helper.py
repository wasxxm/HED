import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import subprocess
from shapely.geometry import Polygon
import os


def iterate_files(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            files.append(os.path.join(directory, filename))
            continue
        else:
            continue
    return files


def display(frame_name, img):
    h, w = img.shape[0:2]
    new_w = 500
    new_h = int(new_w * (h / w))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(frame_name, img)
    cv2.waitKey(0)


def display_org(frame_name, img):
    cv2.imshow(frame_name, img)
    cv2.waitKey(0)


def rotate_img(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def resize_img(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        if height is None:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def cv2pil(src):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil2cv(src):
    return np.asarray(src)


def blur_img_edges(input_img, radius=6):
    im = cv2pil(input_img)

    # Paste image on white background
    diam = 2 * radius
    back = Image.new('RGBA', (im.size[0] + diam, im.size[1] + diam), (120, 120, 120, 0))
    back.paste(im, (radius, radius))

    # Create paste mask
    mask = Image.new('L', back.size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0 = 0, 0
    x1, y1 = back.size
    for d in range(diam + radius):
        x1, y1 = x1 - 1, y1 - 1
        alpha = 255 if d < radius else int(255 * (diam + radius - d) / diam)
        draw.rectangle([x0, y0, x1, y1], outline=alpha)
        x0, y0 = x0 + 1, y0 + 1

    # Blur image and paste blurred edge according to mask
    blur = back.filter(ImageFilter.GaussianBlur(radius / 2))
    back.paste(blur, mask=mask)
    return cv2.cvtColor(pil2cv(back), cv2.COLOR_RGBA2BGRA)


def add_alpha(rgb_data):
    try:
        rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
        # Then assign the mask to the last channel of the image
        rgba[:, :, 3] = 255
        return rgba
    except:
        None


def save_img(src, dest_path):
    # bgr = src[:, :, :3]  # Channels 0..2
    # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Some sort of processing...
    # bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # alpha = src[:, :, 3]  # Channel 3
    # result = np.dstack([bgr, alpha])  # Add the alpha channel
    cv2.imwrite(dest_path, src)


def overlay_transparent(background_src, overlay, x_offset, y_offset):
    background = background_src.copy()
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x_offset >= background_width or y_offset >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x_offset + w > background_width:
        w = background_width - x_offset
        overlay = overlay[:, :w]

    if y_offset + h > background_height:
        h = background_height - y_offset
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y_offset:y_offset + h, x_offset:x_offset + w] = (1.0 - mask) * background[y_offset:y_offset + h,
                                                                              x_offset:x_offset + w] + mask * overlay_image
    return background


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=np.float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)


def img_change_perspective(src_img):
    img = cv2pil(src_img)
    width, height = img.size
    m = -0.5
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE, (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
    return pil2cv(img)


def img_change_perspective2(src_img, distort_upper_left=50, distort_upper_right=50):
    input_img = "temp/input.png"
    output_img = "temp/output.png"
    distort_upper_left = str(distort_upper_left)
    distort_upper_right = str(distort_upper_right)
    save_img(src_img, input_img)
    cmd = "magick " + input_img + " -virtual-pixel transparent +distort Perspective \"0,0,0,0  %[fx:w-1],0,%[fx:w-1],0  0,%[fx:h-1],-" + distort_upper_left + ",%[fx:h-1]  %[fx:w-1],%[fx:h-1],%[fx:w+" + distort_upper_right + "],%[fx:h-1]\" " + output_img
    subprocess.call(cmd, shell=True)
    return cv2.imread(output_img, cv2.IMREAD_UNCHANGED)


def percentage_intersection(img1, img2, offset_x, offset_y):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    box_1 = [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    box_2 = [[0 + offset_x, 0 + offset_y], [w2 + offset_x, 0 + offset_y], [w2 + offset_x, h2 + offset_y],
             [0 + offset_x, h2 + offset_y]]
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_2.area
    return iou


def create_blank_img(width, height, rgb_color=(0, 0, 0)):
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


def add_border_to_img(img, border_width=1, color=[255, 255, 255]):
    top, bottom, left, right = [border_width] * 4
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
