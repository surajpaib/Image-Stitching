# Descriptor type and patch size experiments
results_file='pixel_neighbourhood'
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor pixel_neigbhourhood --patch_size 3 --results_file $results_file
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor pixel_neigbhourhood --patch_size 5 --results_file $results_file
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor pixel_neigbhourhood --patch_size 7 --results_file $results_file

results_file='smooth_pixel_neighbourhood'
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor smooth_pixel_neigbhourhood --patch_size 3 --results_file $results_file
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor smooth_pixel_neigbhourhood --patch_size 5 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor smooth_pixel_neigbhourhood --patch_size 7 --results_file $results_file 

results_file='histogram_pixel_neighbourhood'
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor histogram_pixel_neigbhourhood --patch_size 3 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor histogram_pixel_neigbhourhood --patch_size 5 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor histogram_pixel_neigbhourhood --patch_size 7 --results_file $results_file 

results_file='sift'
python imageStitching.py images/leftImage.png images/rightImage.png --descriptor sift --results_file $results_file 


# Matching method type and top matches experiments
results_file='correlation'
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 100 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 200 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 500 --results_file $results_file 


results_file='euclidean'
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 100 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 200 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 500 --results_file $results_file 

results_file='correlation_sift'
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 100 --results_file $results_file  --descriptor sift
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 200 --results_file $results_file  --descriptor sift
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method correlation --n_matches 500 --results_file $results_file  --descriptor sift


results_file='euclidean_sift'
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 100 --results_file $results_file  --descriptor sift
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 200 --results_file $results_file  --descriptor sift
python imageStitching.py images/leftImage.png images/rightImage.png --matching_method euclidean --n_matches 500 --results_file $results_file  --descriptor sift



# RANSAC Experiments
results_file='ransac_iterations'
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 3 --results_file $results_file --no_gui True 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 5 --results_file $results_file --no_gui True 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 12 --results_file $results_file --no_gui True 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 25 --results_file $results_file --no_gui True 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_iterations 57 --results_file $results_file --no_gui True 

python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 3 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 5 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 12 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 25 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 57 --results_file $results_file  --n_matches 500

results_file='ransac_inlier_threshold'

python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 20 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 50 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 100 --results_file $results_file 
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_inlier_threshold 500 --results_file $results_file 


python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 20 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 50 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 100 --results_file $results_file  --n_matches 500
python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_inlier_threshold 500 --results_file $results_file  --n_matches 500