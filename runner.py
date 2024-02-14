#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, threshold_minimum
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import os
from tqdm import tqdm

def load_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def get_center(image, size=500):
    center = (int(image.shape[0]/2), int(image.shape[1]/2))
    if size > center[0] or size > center[1]:
        print("size too big, returning original image.")
        return image
    return image[center[0]-size : center[0]+size, 
                 center[1]-size : center[1]+size]

def get_sus_areas(gray):
    ## I assume that the center of the image always contains mostly the pole
    ## lets 'zoom in' to the center of the image    
    gray_cneter = get_center(gray)

    # apply threshold
    thresh = threshold_minimum(gray)
    bw = closing(gray_cneter > thresh*0.9, square(13))

    # label image regions
    label_img = label(bw)
    regions = regionprops(label_img)

    # image with labeled regions
    image_label_overlay = label2rgb(label_img, image=gray_cneter)
    return regions, image_label_overlay, thresh

def plot_regions_for_debugging(image, regions):

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    for props in regions:
        if props.axis_major_length < 100:
            continue
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + np.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - np.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - np.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - np.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    plt.show()

def find_angle(regions):
    potential_poles = []
    for props in regions:
        if props.axis_major_length < 100:
            continue
        potential_poles.append(props.orientation)
    if len(potential_poles) == 0:
        return None
    if len(potential_poles) > 1:
        print("More than one pole found, returning the first one")
    return potential_poles[0]

def rotate_image(image, angle):
    ## Rotate img by the angle so the pole is vertical  
    image = image.copy()  
    angle = -angle * 180 / np.pi # convert ang to degrees
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def find_global_x(bw, crop_size, o_img_size):
    # label image regions
    label_img = label(bw)
    regions = regionprops(label_img)

    for props in regions:
        if props.axis_major_length < 100:
            continue
        y0, x0 = props.centroid

        return (int(x0 - crop_size + o_img_size[1]/2), 
                int(y0 - crop_size + o_img_size[0]/2))

def slice_and_dice(res, pole_x, o_img_size, hight=500, width=500):
    ## I want slices about 500 pixels long
    half_width = int(width/2)
    ratio = int(o_img_size[0]/hight)
    y_slice = int(o_img_size[0]/ratio)
    imgs = []
    for i in range(ratio):
        img = res[y_slice * i : y_slice * (i+1),
                pole_x-half_width: pole_x+half_width]
        
        imgs.append(img)
    return imgs

def img_strip(res, pole_x, width=500):
    half_width = int(width/2)
    return res[:, pole_x-half_width: pole_x+half_width]

def plot_multi_imgs(res, pole_x, imgs):
    img_layout = []
    for i in range(len(imgs)):
        img_layout.append([f'{i})'])
    for i in range(len(imgs)):
        img_layout[i].append(f'{len(imgs)})')

    fig, axs = plt.subplot_mosaic(img_layout, layout='constrained')

    for label, ax in axs.items():
        i = int(label.strip(")"))
        if i>=len(imgs):
            ax.imshow(img_strip(res, pole_x, width=500), cmap='gray')
        else:
            ax.imshow(imgs[i], cmap='gray')
        ax.set_axis_off()
    plt.show()

def run_along_pole(img, debug = False):
    gray = load_image(img)

    regions, image_label_overlay, thresh = get_sus_areas(gray)

    pole_angle = find_angle(regions)

    res = rotate_image(gray, pole_angle)

    o_img_size = res.shape
    crop_size = 500

    res_cneter = get_center(res, crop_size)

    bw = closing(res_cneter > thresh*0.9, square(13))

    pole_x, pole_y = find_global_x(bw, crop_size, o_img_size)

    o_res = rotate_image(img, pole_angle)
    imgs = slice_and_dice(o_res, pole_x, o_img_size, hight=300, width=500)
    if debug:
        plot_regions_for_debugging(image_label_overlay, regions)
        
        plot_multi_imgs(res, pole_x, imgs)
    
    return imgs

cwd = os.getcwd()
folder_path = os.path.join(cwd, r"imgs\side_sample")
img_lst = os.listdir(folder_path)

## make a results folder
results_folder = os.path.join(folder_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

for path in img_lst:
    if path.endswith(".jpg") or path.endswith(".JPG"):
        full_path = os.path.join(folder_path, path)
        img = cv2.imread(full_path)
        imgs = run_along_pole(img, True)
        ## save the images to the results folder
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(results_folder, f"{path}_{i}.jpg"), img)

# %%
