conda activate conda-env

python3 src/main.py

The idea was to map the 8 corners of a crop box to the 8 corners of a new matrix using a affine transformation. 

Classes were coded to load the 3d images and file data from the folders. 
Additionally, the get corners function gets the 8 corners of the crop box and the same method is used to get the 8 corners of the output 3d image to ensure consistency
the get affine transformation from points should produce the correct output but i couldn't get the affine transformation to produce a non-black image