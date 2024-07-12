#! /bin/bash

etas1=(
    "0.2"
    "1"
    "2"
    "3"
)
for eta in "${etas1[@]}"; do
    echo "Working on eta = $eta, scattered"
    
    tag="scatter"
    julia axis_graph.jl $tag $eta
done

etas2=(
    "0.2"
    "3"
)
for eta in "${etas2[@]}"; do
    echo "Working on eta = $eta, grid"
    
    tag="grid"
    julia axis_graph.jl $tag $eta
done

