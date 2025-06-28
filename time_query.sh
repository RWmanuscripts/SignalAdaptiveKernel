#! /bin/bash

# # vary down_factor
julia time_query_1.jl 2
julia time_query_1.jl 4
julia time_query_1.jl 6
julia time_query_1.jl 8
julia time_query_1.jl 10

# vary b_x
julia time_query_lazy.jl 6
julia time_query_lazy.jl 8
julia time_query_lazy.jl 10
julia time_query_lazy.jl 12
julia time_query_lazy.jl 14

