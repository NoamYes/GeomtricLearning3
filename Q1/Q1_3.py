import numpy as np
from scipy.signal import convolve2d
import cv2
from PIL import Image
from Q1.shortest_paths import eikonal_path_grad
import numpy as np
import matplotlib.pyplot as plt
import eikonalfm


def compute_geodesic(c_field, p1, p2):
    dx = (1, 1)
    order = 2
    tau = eikonalfm.fast_marching(c_field, p1, dx, order)  # TODO
    path_len = tau[p2]-tau[p1]
    path_indices, path_points = eikonal_path_grad(tau, p1, p2)
    tau_vals = np.array([tau[tuple(path_indices[int(i)])] for i in range(np.shape(path_indices)[0])])
    midway_point = path_indices[np.argmin(abs(tau_vals - path_len / 2)), :]
    return path_indices, path_points, path_len, midway_point


def geodesic_image_segmentation(img, sigma, ksize, init_points):

    (dimY, dimX, dimC) = np.shape(img)
    gaus_kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    gaus_kernel = gaus_kernel.dot(gaus_kernel.T)
    gx_gaus_kernel, gy_gaus_kernel = np.gradient(gaus_kernel)
    grads_image = []
    for i in range(dimC):
        img_C = img[:, :, i]
        gx_image = convolve2d(img_C, gx_gaus_kernel, 'same')
        gy_image = convolve2d(img_C, gy_gaus_kernel, 'same')
        grads_image = grads_image + [gx_image, gy_image]

    grads_arrays = np.stack(grads_image, axis=2)
    g_image = 1 / (1 + np.linalg.norm(grads_arrays, axis=2))

    plt.figure(1)
    plt.imshow(g_image, cmap='jet')
    plt.show()

    curr_points = np.copy(init_points)
    next_points = np.zeros(np.shape(init_points))

    epsilon = 1e-2
    diff_eps = epsilon*1e4  # TODO
    next_eps = diff_eps
    diff_eps_list = [diff_eps]
    while diff_eps > epsilon:
        curr_eps = next_eps
        next_eps = 0
        GAC = np.empty((0, 2))
        for i in range(4):
            p1 = tuple(curr_points[i].astype('int'))
            p2 = tuple(curr_points[(i + 1) % 4].astype('int'))
            geo_indices, geo_points, path_len, midway_point = compute_geodesic(g_image**0.8, p1, p2)
            next_points[i] = midway_point
            next_eps = next_eps + path_len
            GAC = np.vstack([GAC, geo_indices])
        curr_points = next_points
        diff_eps = abs(curr_eps-next_eps)
        image_file_path = np.copy(duck_img)
        plt.imshow(image_file_path, cmap='gray')
        plt.scatter(GAC[:, 1], GAC[:, 0], s=3, c='green')
        plt.show()
    return GAC


# Load image to ndarray
image_file_original = Image.open("./duck.png")  # open colour image
# image_file_original = Image.open("./present.png")  # open colour image
# image_file_original = Image.open("./some_ball.jpg")  # open colour image
# image_file_original = Image.open("./japan.png")  # open colour image
# image_file_original = Image.open("./israel.png")  # open colour image
image_file = image_file_original.convert("RGB")
duck_img = np.asarray(image_file)

dimX = np.shape(duck_img)[1]
dimY = np.shape(duck_img)[0]

sigma = 10
ksize = 10

init_points = np.zeros((4, 2))
init_points[0] = [0, 0]
init_points[1] = [0, dimX - 1]
init_points[2] = [dimY - 1, dimX - 1]
init_points[3] = [dimY - 1, 0]

geodesic_image_segmentation(duck_img, sigma, ksize, init_points)
print('ya')