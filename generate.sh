results_file='window_size'
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --harris_neighbourhood_size 2 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --harris_neighbourhood_size 5 --no_gui True --wandb True
python imageStitching.py images/leftImage.png images/rightImage.png --results_file $results_file --harris_neighbourhood_size 7 --no_gui True --wandb True

python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --harris_neighbourhood_size 2 --no_gui True --wandb True
python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --harris_neighbourhood_size 5 --no_gui True --wandb True
python imageStitching.py images/leftim.png images/rightim.png --results_file $results_file --harris_neighbourhood_size 7 --no_gui True --wandb True

python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --harris_neighbourhood_size 2 --no_gui True --wandb True
python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --harris_neighbourhood_size 5 --no_gui True --wandb True
python imageStitching.py images/im1.jpg images/im2.jpg --results_file $results_file --harris_neighbourhood_size 7 --no_gui True --wandb True
