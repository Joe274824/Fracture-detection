import numpy as np

# Crop box coordinates and sizes
center = [-12.522502899169924, -132.86927795410156, -507.9060974121094]
orientation = [-0.995144660117605, -0.06923995523797782, -0.06994951063488068, 
               0.0690695362340799, -0.9976000343818373, 0.0048549527048470315, 
               -0.07011779092231578, 9.757666536167155e-19, 0.9975387187453801]
size = [119.89214361207402, 148.5603213552619, 199.2863756483113]

# Metadata of the 3D image
dimension = 3
space = 'left-posterior-superior'
sizes = [512, 512, 665]
space_directions = [(0.771484375,0,0), (0,-0.771484375,0), (0,0,1)]
kinds = ['domain', 'domain', 'domain']
encoding = 'gzip'
space_origin = (-158.1142578125, 65.1142578125, -632)

# Calculate crop box parameters
half_size = [x/2 for x in size]
corner = [c - hs for c, hs in zip(center, half_size)]
orientation_matrix = np.reshape(orientation, (3, 3))
directions = np.array(space_directions)

# Compute crop box corners in image space
c1 = corner[0] * directions[0] + corner[1] * directions[1] + corner[2] * directions[2]
c2 = c1 + size[0] * directions[0]
c3 = c1 + size[0] * directions[0] + size[1] * directions[1]
c4 = c1 + size[1] * directions[1]
c5 = c1 + size[2] * directions[2]
c6 = c1 + size[0] * directions[0] + size[2] * directions[2]
c7 = c1 + size[0] * directions[0] + size[1] * directions[1] + size[2] * directions[2]
c8 = c1 + size[1] * directions[1] + size[2] * directions[2]

# Convert corner points to image indices
c1_idx = np.rint((c1 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c2_idx = np.rint((c2 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c3_idx = np.rint((c3 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c4_idx = np.rint((c4 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c5_idx = np.rint((c5 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c6_idx = np.rint((c6 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c7_idx = np.rint((c7 - np.array(space_origin)) / np.array(space_directions)).astype(int)
c8_idx = np.rint((c8 - np.array(space_origin)) / np.array(space_directions)).astype(int)

# Crop the region from the 3D image using the computed indices
# crop_array = original_image[c1_idx[0]:c7_idx[0], c1_idx[1]:c7_idx[1], c1_idx[2]:c7_idx[2]]
print(c1_idx, c7_idx)