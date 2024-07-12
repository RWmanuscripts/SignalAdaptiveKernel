#! /bin/bash

julia setup_packages.jl

# Common settings.
N_workers=7

down_factor=2
#down_factor=4

hopt_save_dir="results/kodak/down$down_factor"
save_dir="$hopt_save_dir/spline32/"

# process each image
for i in {1..24}
do
    echo "Working on image " $i
    
    image_file_name="kodim0$i.png"
    if [ $i -ge 10 ]; then
        image_file_name="kodim$i.png"
    fi
    
    model_selection="ML"
    julia kodak.jl $save_dir $image_file_name $N_workers $model_selection $down_factor
done

julia kodak_results.jl