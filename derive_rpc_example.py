import cv2 as cv
import numpy as np
from rpcfit import rpc_fit

# calculate transformation image to world transformation of image

image_col = 11752
image_row = 10797

srcTri = np.array( [[0, 0], 
                    [image_col - 1, 0],
                    [0, image_row - 1]] ).astype(np.float32)
dstTri = np.array( [[103.731393565315, 1.25799910208483], 
                    [103.679665801256, 1.24738065344267], 
                    [103.721666475672, 1.30587199896675]] ).astype(np.float32)
warp_mat = cv.getAffineTransform(srcTri, dstTri)

print(warp_mat)


pixel = np.array( [[image_col - 1], [image_row - 1], [1]] ).astype(np.float32)
world = np.matmul(warp_mat, pixel)
print(pixel)
print(world)

# create a meshgrid of at least 25 x 25 corresponding tie-points from 2D image space to 3D world space
# 3D world space can have flat terrain. However flat terrain must be representative by random small numbers so that RPC can be solved.

dimension_size = 25
x_grid, y_grid = np.meshgrid(np.linspace(0, image_col, dimension_size), np.linspace(0, image_col, dimension_size))
z_grid = np.full((dimension_size, dimension_size), 1)
positions = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

image_positions = np.transpose(np.vstack([x_grid.ravel(), y_grid.ravel()]))

world_positions = np.transpose(np.matmul(warp_mat, positions))
z_random_positions = np.random.random_sample(dimension_size * dimension_size) / 1000000.0
world_3d_positions = np.insert(world_positions, 2, z_random_positions, axis = 1)

# fit on training set
rpc_calib, log = rpc_fit.calibrate_rpc( image_positions, world_3d_positions, separate=False, tol=1e-10
                                      , max_iter=100, method='initLcurve'
                                      , plot=True, orientation = 'projloc', get_log=True )


print(rpc_calib)

# Evaluate on training set
rmse_err, mae, planimetry = rpc_fit.evaluate(rpc_calib, world_3d_positions, image_positions)
print('Training set :   Mean X-RMSE {:e}     Mean Y-RMSE {:e}'.format(*rmse_err))
