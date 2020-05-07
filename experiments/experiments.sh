source ../cvenv/bin/activate

patch_sizes=(3 5 7)

for descriptor in 'sift' 'pixel_neighbourhood'
do
    for matching_method in 'euclidean' 'correlation'
    do
        if (( $descriptor == 'pixel_neighbourhood' ))
        then
            for patch_size in "${patch_sizes[@]}"
            do 
                python imageStitching.py images/leftim.png images/rightim.png --descriptor $descriptor --matching_method $matching_method --patch_size $patch_size --wandb True --no_gui True
            done

        else
            python imageStitching.py images/leftim.png images/rightim.png --descriptor $descriptor --matching_method $matching_method --wandb True --no_gui True
        fi
    done
done
