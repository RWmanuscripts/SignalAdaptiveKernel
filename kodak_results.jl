# collects the results after running new.sh.
using Pkg
Pkg.activate(".")

using LinearAlgebra, DelimitedFiles, Serialization

import Images
using FileIO, ImageMagick

include("helpers/image.jl")
include("helpers/hopt.jl")
include("helpers/kodak_tables.jl")

T = Float64

reference_image_folder = "data/images/kodak/"

rs = 1:6

down_factor = 2
results_dir = "results/kodak/down$(down_factor)/spline32/"

hps, SSIMs_itp, SSIMs_sk, SSIMs_deks = process_kodak_results(
    T,
    reference_image_folder,
    results_dir;
    transform = "none",
    scenes = 1:24,
    model_select_string = "ML",
)

hp_mat = generate_hp_table(hps; sig_digits = 3)
writedlm( "tables/kodak_hp.csv",  hp_mat, ',')

writedlm( "tables/ssim_itp.csv",  SSIMs_itp, ',')
writedlm( "tables/ssim_sk.csv",  SSIMs_sk, ',')
writedlm( "tables/ssim_de.csv",  SSIMs_deks, ',')

### generate tables.
SSIM_mat = generate_SSIM_table(
    SSIMs_itp, SSIMs_sk, SSIMs_deks;
    sig_digits = 5
)


# # Assemble manuscript table

scene_names = get_scene_names()

# # Assemble summary table
using Markdown, Tables, MarkdownTables

DEK_r_header = collect(
    "DEK r: $r" for r in rs
)

data_mat = [
    scene_names SSIM_mat
];

print(markdown_table(Tables.table(
    data_mat;
    header = [
        "Scene"; "Bi-cubic"; "CK"; DEK_r_header;
    ]
), String))


DEK_hp_header = collect(
    "Gain, r: $r" for r in rs
)
print(markdown_table(Tables.table(
    [scene_names hp_mat];
    header = [
        "Scene"; "Bandwidth"; DEK_hp_header;
    ]
), String))

nothing