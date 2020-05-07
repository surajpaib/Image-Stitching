# Best parameters for different weather stitching. Section 5.1
python imageStitching.py images/im1.jpg images/im3.jpg --descriptor sift --RANSAC_inlier_threshold 50 --n_matches 1000 --matching_method euclidean --harris_keypoint_threshold 0.01 --harris_neighbourhood_size 2


# Best parameters for 3 image stitching. Section 5.2
python multiImageStitching.py images/1.jpeg images/2.jpeg images/3.jpeg

# Best parameters for historical stitching. Section 5.3
python imageStitching.py images/old_1.png images/new_1.png --descriptor sift --RANSAC_inlier_threshold 5 --RANSAC_iterations 5000 --harris_keypoint_threshold 0.01 --harris_neighbourhood_size 2 --matching_method euclidean

python imageStitching.py images/old2.png images/new2.png 

# Best parameters for 4 season stitching. Section 5.4
python multiImageStitching.py images/season1.png images/season2.png images/season3.png images/season4.png --RANSAC_inlier_threshold 10 --RANSAC_iterations 500 --n_matches 1000