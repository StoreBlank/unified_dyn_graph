import numpy as np
import cv2


img_path = '/home/baoyu/2023/unified_dyn_graph/data_dense/mustard_bottle/camera_1/episode_0/0_color.png'
extr = np.load('/home/baoyu/2023/unified_dyn_graph/data_dense/mustard_bottle/camera_1/camera_extrinsic_matrix.npy')
intr = np.load('/home/baoyu/2023/unified_dyn_graph/data_dense/mustard_bottle/camera_1/camera_intrinsic_params.npy')
# u = [ 0.3713784, 1.37706299, -0.08058137, 0.48177189]
u = [1., 1., 0., 0.]


fx, fy, cx, cy = intr

p = np.array([[u[0], 0.5, -u[1]], [u[2], 0.5, -u[3]]])

p_homo = np.concatenate([p, np.ones((2, 1))], axis=1)

p_cam = p_homo @ extr.T

p_cam[:, 1] *= -1
p_cam[:, 2] *= -1

p_projs = np.zeros((2, 2))

p_projs[:, 0] = p_cam[:, 0] * fx / p_cam[:, 2] + cx
p_projs[:, 1] = p_cam[:, 1] * fy / p_cam[:, 2] + cy

img = cv2.imread(img_path)
colors = [(0, 0, 255), (0, 255, 0)]  # start: red, end: green
for i in range(2):
    cv2.circle(img, (int(p_projs[i, 0]), int(p_projs[i, 1])), 3, colors[i], -1)


p = np.array([[u[0], u[1], 0], [u[0]+1, u[1], 0], [u[0], u[1] + 1, 0], [u[0], u[1], 1]])

p_homo = np.concatenate([p, np.ones((4, 1))], axis=1)

p_cam = p_homo @ extr.T

p_cam[:, 1] *= -1
p_cam[:, 2] *= -1

p_projs = np.zeros((4, 2))

p_projs[:, 0] = p_cam[:, 0] * fx / p_cam[:, 2] + cx
p_projs[:, 1] = p_cam[:, 1] * fy / p_cam[:, 2] + cy


cv2.line(img,(int(p_projs[0, 0]), int(p_projs[0, 1])), (int(p_projs[1, 0]), int(p_projs[1, 1])), (0,0,255), 1)

cv2.line(img,(int(p_projs[0, 0]), int(p_projs[0, 1])), (int(p_projs[2, 0])+1, int(p_projs[2, 1])), (0,255,0), 1)
cv2.line(img,(int(p_projs[0, 0]), int(p_projs[0, 1])), (int(p_projs[3, 0])+1, int(p_projs[3, 1])), (255,0,0), 1)

cv2.imwrite('point_visualize.jpg', img)
