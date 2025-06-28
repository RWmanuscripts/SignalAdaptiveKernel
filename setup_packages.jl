using Pkg
Pkg.activate(".")
Pkg.Registry.add(RegistrySpec(url = "https://github.com/RoyCCWang/RWPublicJuliaRegistry"))
let
    pkgs = [
        "DelimitedFiles", "JSON3", "PlotlyLight", "DataFrames", "CSV", "SingleLinkagePartitions", "LazyGPR", "SpatialGSP", "Interpolations", "ScatteredInterpolation", "PythonPlot", "Revise", "IJulia", "Images", "RieszDSP", "LocalFilters", "VisualizationBag", "StaticArrays",
        "Markdown", "Tables", "MarkdownTables", "FileIO",
    ]
    for pkg in pkgs
        if isnothing(Base.find_package(pkg))
            Pkg.add(pkg)
        end
    end
end

nothing
