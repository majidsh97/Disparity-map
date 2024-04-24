import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import medfilt
from ex2 import extract_features, filter_and_align_descriptors

# These are type hints, they mostly make the code readable and testable
t_img = np.array
t_disparity = np.array


def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """finds the maximum translation/shift between two images
    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y wise distance between valid matching points.

    Returns:
        an integer value representing the maximum translation of the camera from src to dst image
    """
    # Step 1: Generate features/descriptors, filter and align them (courtesy of exercise 2)
    src_features = extract_features(src)
    dst_features = extract_features(dst)
    src_pts, dst_pts = filter_and_align_descriptors(src_features, dst_features)
    # Step 2: filter out correspondences that are not horizontally aligned using well aligned threshold
    filtered_points = []
    for i in range(len(src_pts)):
        src_x, src_y = src_pts[i]
        dst_x, dst_y = dst_pts[i]
        if abs(src_y - dst_y) < well_aligned_thr:
            filtered_points.append(((src_x, src_y), (dst_x, dst_y)))
    # Step 3: Find the translation across the image using the descriptors and return the maximum value
    translations = [aligned_point_dst[0] - aligned_point_src[0] for aligned_point_src, aligned_point_dst in
                    filtered_points]
    # Calculate the maximum translation in the x direction
    max_abs_idx = np.argmax(np.abs(translations))
    # Get the corresponding element
    largest_with_sign = translations[max_abs_idx]
    max_translation = int(np.round(largest_with_sign))
    return max_translation


def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.
    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """
    # Step 1: Pad necessary values to src and dst
    padded_src = np.pad(src, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    padded_dst = np.pad(dst, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

    # Step 2: find the disparity value and return
    height, width, _ = src.shape
    disparity = np.zeros((height, width))

    for y in range(pad_size, height + pad_size):
        for x in range(pad_size, width + pad_size):
            shifted_pixel = padded_src[y, x + offset]
            dst_pixel = padded_dst[y, x]

            # Calculate Euclidean distance between RGB values
            distance = np.linalg.norm(shifted_pixel - dst_pixel)
            disparity[y - pad_size, x - pad_size] = distance

    return disparity


def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images
    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter
    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """
    # Step 1: Construct a stack of all reasonable disparity hypotheses.
    disparity_stack = np.stack([
        render_disparity_hypothesis(src, dst, o, pad_size)
        for o in range(-offset, offset + 1)
    ])
    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter onto
    # the stack of disparity hypotheses
    smoothed_disparities = gaussian_filter(disparity_stack, [sigma_z, sigma_x, 0])
    # Step 3: Choose the best disparity hypothesis for every pixel
    best_disparity = np.argmin(smoothed_disparities, axis=0) + (-offset)  # should we add offset

    # Step 4: Apply the median filter to enhance local consensus
    # apply    # a # median    # filter    # to    # enhance    # local    # consensus(similar    # to    # removing
    # salt - pepper    # noise) from the disparity    # map.
    final_disparity_map = medfilt(best_disparity, kernel_size=median_filter_size)
    return final_disparity_map


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates src x-direction
        y_array: a numpy array of [H' x W'] representing interpolation in y-direction

    Returns:
        An image of size [H' x W'] containing the sampled points in
    """

    # Step 1: Estimate the left, top, right, bottom integer parts (l, r, t, b)
    # and the corresponding coefficients (a, b, 1-a, 1-b) of each pixel
    l = np.floor(x_array).astype(int)
    t = np.floor(y_array).astype(int)
    r = l + 1
    bo = t + 1

    a = x_array - l
    b = y_array - t
    a_inv = 1 - a
    b_inv = 1 - b

    # Step 2: Take care of out of image coordinates
    max_height, max_width, _ = img.shape
    l = np.clip(l, 0, max_width - 1)
    r = np.clip(r, 0, max_width - 1)
    t = np.clip(t, 0, max_height - 1)
    bo = np.clip(bo, 0, max_height - 1)

    # Step 3: Produce a weighted sum of each rounded corner of the pixel
    top_left = img[t, l]
    top_right = img[t, r]
    bottom_left = img[bo, l]
    bottom_right = img[bo, r]

    weighted_top_left = top_left * (a_inv * b_inv)[:, :, np.newaxis]
    weighted_top_right = top_right * (a * b_inv)[:, :, np.newaxis]
    weighted_bottom_left = bottom_left * (a_inv * b)[:, :, np.newaxis]
    weighted_bottom_right = bottom_right * (a * b)[:, :, np.newaxis]
    # Step 4: Accumulate and return all the weighted four corners

    sampled_img = weighted_top_left + weighted_top_right + weighted_bottom_left + weighted_bottom_right

    return sampled_img

#
