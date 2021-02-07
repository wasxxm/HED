from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import cv2
import numpy as np
from configs import *


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
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

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
    back = Image.new('RGBA', (im.size[0] + diam, im.size[1] + diam), (255, 255, 255, 0))
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
    return pil2cv(back)


def add_alpha(rgb_data):
    rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
    # Then assign the mask to the last channel of the image
    rgba[:, :, 3] = 255
    return rgba


def save_img(src, dest_path):
    # bgr = src[:, :, :3]  # Channels 0..2
    # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Some sort of processing...
    # bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # alpha = src[:, :, 3]  # Channel 3
    # result = np.dstack([bgr, alpha])  # Add the alpha channel
    cv2.imwrite(dest_path, src)


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
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

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image
    return background
