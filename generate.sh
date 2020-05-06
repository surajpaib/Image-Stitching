results_file='correlation'
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 100 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 200 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 500 --no_gui True --wandb True


results_file='correlation_sift'
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 100 --descriptor sift --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 200 --descriptor sift --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method correlation --n_matches 500 --descriptor sift --no_gui True --wandb True


# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor pixel_neighbourhood --patch_size 7 --no_gui True --wandb True

# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor pixel_neighbourhood --patch_size 7 --no_gui True --wandb True



results_file='euclidean'
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method euclidean --n_matches 100 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method euclidean --n_matches 200 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --matching_method euclidean --n_matches 500 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 7 --no_gui True --wandb True

# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor smooth_pixel_neighbourhood --patch_size 7 --no_gui True --wandb True


# results_file='histogram_pixel_neighbourhood_patch_size'
# python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 7 --no_gui True --wandb True

# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 7 --no_gui True --wandb True

# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 3 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 5 --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor histogram_pixel_neighbourhood --patch_size 7 --no_gui True --wandb True



# results_file='sift'
# python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --descriptor sift --no_gui True --wandb True

# python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --descriptor sift --no_gui True --wandb True

# python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --descriptor sift --no_gui True --wandb True
