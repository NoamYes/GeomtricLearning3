import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 

thresh = 150 # TODO What value should it be?
image_file = Image.open("../HW3_Resources/maze.png") # open colour image
image_file = image_file.convert("L")
maze_img = np.asarray(image_file).astype('int')
maze_img = maze_img >= thresh
maze_img = maze_img.astype('int')

# Remove all 1 lines and columns from maze
maze_img_cpy = np.copy(maze_img)
maze_img[np.where(maze_img_cpy.all(1)), :] = 0 
maze_img[:, np.where(maze_img_cpy.all(0))] = 0

plt.imshow(maze_img, cmap='gray', vmin=0, vmax=1)
# plt.show()



src_point = (383, 814)

# Init T0 matrix and colors labels
T0 = np.empty(np.shape(maze_img))
T0[:,:] = float('inf')
T0[src_point] = 0

# Init F(x,y) 
F = np.copy(maze_img)
dimY = np.shape(maze_img)[0]
dimX = np.shape(maze_img)[1]

# Define each step iteration calculation

def step(T, F):
    T_cpy = np.copy(T)
    for i in np.arange(dimY):
        for j in np.arange(dimX):
            if T_cpy[i,j] == float('inf'):
                T_north = T_cpy[i-1,j] if i > 0 else float('inf')
                T_south = T_cpy[i+1,j] if i < dimY-1 else float('inf')
                T_east = T_cpy[i,j-1] if j > 0 else float('inf')
                T_west = T_cpy[i,j+1] if j < dimX-1 else float('inf')
                T1 = min(T_north, T_south)
                T2 = min(T_east, T_west)
                if abs(T1-T2) < F[i,j]:
                    T[i,j]=(T1+T2+np.sqrt(2*F[i,j]**2-(T1-T2)**2))/2
                else:
                    T[i,j] = min(T1,T2) + F[i,j]
    return T
            

# iterate untill convergence
threshold = 1e-2
T = T0
break_condition = False
iter = 0
while not break_condition:
    iter = iter + 1
    T_next = step(T,F)
    T = T_next
    plt.imshow(T == float('inf'))
    plt.show()


