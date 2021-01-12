import numpy as np
from scipy.signal import convolve2d
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import eikonalfm
import gif as gif

from utils.shortest_paths import eikonal_path_grad

gif.options.matplotlib["dpi"] = 300

def compute_geodesic(c_field, p1, p2):
    dx = (1, 1)
    order = 2
    tau = eikonalfm.fast_marching(c_field, p1, dx, order)  # TODO
    path_len = tau[p2]-tau[p1]
    path_indices, path_points = eikonal_path_grad(tau, p1, p2)
    tau_vals = np.array([tau[tuple(path_indices[int(i)])] for i in range(np.shape(path_indices)[0])])
    midway_point = path_indices[np.argmin(abs(tau_vals - path_len / 2)), :]
    return path_indices, path_points, path_len, midway_point



def geodesic_image_segmentation(img, sigma, ksize, init_points, power):

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
    g_image = 1 / (1 + np.linalg.norm(grads_arrays, axis=2))**power

    plt.figure(1)
    plt.imshow(g_image, cmap='jet')
    plt.show()

    curr_points = np.copy(init_points)
    next_points = np.zeros(np.shape(init_points))

    epsilon = 1e-2
    diff_eps = epsilon*1e4  # TODO
    next_eps = diff_eps
    diff_eps_list = [diff_eps]
    iter = 0
    frames = []
    GAC_list = []
    while diff_eps > epsilon and iter < 20:
        iter = iter + 1
        curr_eps = next_eps
        next_eps = 0
        GAC = np.empty((0, 2))
        for i in range(4):
            p1 = tuple(curr_points[i].astype('int'))
            p2 = tuple(curr_points[(i + 1) % 4].astype('int'))
            geo_indices, geo_points, path_len, midway_point = compute_geodesic(g_image, p1, p2)
            next_points[i] = midway_point
            next_eps = next_eps + path_len
            GAC = np.vstack([GAC, geo_indices])
        curr_points = next_points
        diff_eps = abs(curr_eps-next_eps)
        GAC_list.append(GAC)
    return GAC_list, frames


# Load image to ndarray
# fileName = 'duck.png' # power = 5 and sigma=10 ksize=10
# fileName = 'present.png' 
# fileName = 'some_ball.jpg' # power=1 sigma 10 ksize = 7
# fileName = 'orange.jpeg' # power=2 sigma 20 ksize = 10
fileName = 'pineapple.jpg' # power=1 sigma 20 ksize = 20
# fileName = 'japan.png' # power=4 sigma 10 ksize = 7
# fileName = 'Israel.png' # power=1 sigma 10 ksize = 7


image_file_original = Image.open("./Q1/imgs/" + fileName)  # open colour image
image_file = image_file_original.convert("RGB")
duck_img = np.asarray(image_file)

dimX = np.shape(duck_img)[1]
dimY = np.shape(duck_img)[0]

sigma = 20
ksize = 20
power = 4

init_points = np.zeros((4, 2))
init_points[0] = [0, 0]
init_points[1] = [0, dimX - 1]
init_points[2] = [dimY - 1, dimX - 1]
init_points[3] = [dimY - 1, 0]

GAC_list, frames = geodesic_image_segmentation(duck_img, sigma, ksize, init_points, power)

fig3 = plt.figure(3)
im_show = plt.imshow(duck_img, cmap='gray')
scat = plt.scatter(0, 0, s=3, c='green')

def initFunc():
    scat = plt.scatter(0, 0, s=3, c='green')
    return scat, 

def updatefig(j, *fargs):
    global scat
    scat.remove()
    GAC_list = fargs
    GAC = GAC_list[j]
    scat = plt.scatter(GAC[:, 1], GAC[:, 0], s=3, c='green')
    return (scat, )

fargs = GAC_list

ani = matplotlib.animation.FuncAnimation(fig3, updatefig, fargs=fargs, 
                              frames=range(len(GAC_list)), interval=10, blit=True)
path_gifs = "./Q1/gifs/"
ani.save(path_gifs + fileName + '.gif', writer = 'imagemagick', fps = 3) 

