

import numpy as np

def generate_affine_matrix(input_points, destination_points):
    # Ensure points are numpy arrays
    input_points = np.array(input_points)
    destination_points = np.array(destination_points)
    
    # Add homogeneous coordinates (1) to make them 4x4 matrices
    input_points_homogeneous = np.hstack([input_points, np.ones((input_points.shape[0], 1))])
    destination_points_homogeneous = np.hstack([destination_points, np.ones((destination_points.shape[0], 1))])
    
    # Compute affine transformation matrix using least squares
    affine_matrix, res, rank, s = np.linalg.lstsq(input_points_homogeneous, destination_points_homogeneous, rcond=None)
    
    # Make sure the last row is [0, 0, 0, 1] to ensure it's an affine matrix
    affine_matrix[-1] = [0, 0, 0, 1]
    
    return affine_matrix



def getAffineTransformFromPoints(ins, out):
    return generate_affine_matrix(ins[1:5], out[1:5])

    ins = ins[1:5]
    out = out[1:5]
    # calculations
    l = len(ins)
    B = np.vstack([np.transpose(ins), np.ones(l)])
    D = 1.0 / (np.linalg.det(B) + 0.0000000000000001)
    entry = lambda r,d: np.linalg.det(np.delete(np.vstack([r, B]), (d+1), axis=0))
    M = [[(-1)**i * D * entry(R, i) for i in range(l)] for R in np.transpose(out)]
    A, t = np.hsplit(np.array(M), [l-1])
    t = np.transpose(t)[0]

    return A, t
    
    

if __name__ == "__main__":
    ins = [[1, 1, 2], [2, 3, 0], [3, 2, -2], [-2, 2, 3]]  # <- points
    out = [[0, 2, 1], [1, 2, 2], [-2, -1, 6], [4, 1, -3]] # <- mapped to

    # affine, translation = getAffineTransformFromPoints(ins, out)

    print(generate_affine_matrix(ins, out))

    # output
    # print("Affine transformation matrix:\n", affine)
    # print("Affine transformation translation vector:\n", translation)
    # # unittests
    # print("TESTING:")

    # for p, P in zip(np.array(ins), np.array(out)):
    #     image_p = np.dot(affine, p) + translation
    #     result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
    #     print(p, " mapped to: ", image_p, " ; expected: ", P, result)