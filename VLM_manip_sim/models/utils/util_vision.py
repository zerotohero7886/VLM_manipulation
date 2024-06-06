import numpy as np

def compute_xyz(depth_img, camera_info):

    # x, fy, px, py
    fx = camera_info[0]
    fy = camera_info[1]
    cx = camera_info[2]
    cy = camera_info[3]

    height = depth_img.shape[0]
    width = depth_img.shape[1]

    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    
    z_e = depth_img * 0.001 # Convert to meters
    x_e = (indices[..., 1] - cx) * z_e / fx 
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img
