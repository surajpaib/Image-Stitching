# results_file='correlation'
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 100 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 200 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 500 --results_file $results_file --no_gui True --wandb True


# results_file='euclidean'
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 100 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 200 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 500 --results_file $results_file --no_gui True --wandb True

# results_file='correlation_sift'
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 100 --results_file $results_file --no_gui True --wandb True --descriptor sift
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 200 --results_file $results_file --no_gui True --wandb True --descriptor sift
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 500 --results_file $results_file --no_gui True --wandb True --descriptor sift


# results_file='euclidean_sift'
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 100 --results_file $results_file --no_gui True --wandb True --descriptor sift
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 200 --results_file $results_file --no_gui True --wandb True --descriptor sift
# python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 500 --results_file $results_file --no_gui True --wandb True --descriptor sift




# results_file='ransac_iterations_3'
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 3 --results_file $results_file --no_gui True 
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 5 --results_file $results_file --no_gui True 
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 12 --results_file $results_file --no_gui True 
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 25 --results_file $results_file --no_gui True 
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 57 --results_file $results_file --no_gui True 

# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 3 --results_file $results_file --no_gui True --wandb True --n_matches 500
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 5 --results_file $results_file --no_gui True --wandb True --n_matches 500
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 12 --results_file $results_file --no_gui True --wandb True --n_matches 500
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 25 --results_file $results_file --no_gui True --wandb True --n_matches 500
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 57 --results_file $results_file --no_gui True --wandb True --n_matches 500
# results_file='ransac_iterations'
# python imageStitching.py images/974-1.jpg images/975-1.jpg --RANSAC_iterations 50 --results_file $results_file --no_gui True --n_matches 1000
# python imageStitching.py images/974-1.jpg images/975-1.jpg --RANSAC_iterations 100 --results_file $results_file --no_gui True --n_matches 1000
# python imageStitching.py images/974-1.jpg images/975-1.jpg --RANSAC_iterations 1000 --results_file $results_file --no_gui True --n_matches 1000
# python imageStitching.py images/974-1.jpg images/975-1.jpg --RANSAC_iterations 5000 --results_file $results_file --no_gui True --n_matches 1000


# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 2000 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 5000 --results_file $results_file --no_gui True --wandb True

# results_file='ransac_init'
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 3 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 4 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 5 --results_file $results_file --no_gui True --wandb True


results_file='ransac_iterations'


python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 3 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 5 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 12 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 25 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 500 --results_file $results_file --no_gui True --wandb True


python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 3 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 5 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 12 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 25 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 500 --results_file $results_file --no_gui True --wandb True --n_matches 500
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 20 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 50 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/leftImage.png --RANSAC_inlier_threshold 100 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 500 --results_file $results_file --no_gui True --wandb True

results_file='ransac_inlier_threshold'

python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 20 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 50 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 100 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 500 --results_file $results_file --no_gui True --wandb True


python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 20 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 50 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 100 --results_file $results_file --no_gui True --wandb True --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 500 --results_file $results_file --no_gui True --wandb True --n_matches 500