results_file='ransac_init'
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 3 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 4 --results_file $results_file --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --RANSAC_init_points 5 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 2000 --results_file $results_file --no_gui True --wandb True
# python imageStitching.py images/im1.jpg images/im2.jpg --RANSAC_iterations 5000 --results_file $results_file --no_gui True --wandb True


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
