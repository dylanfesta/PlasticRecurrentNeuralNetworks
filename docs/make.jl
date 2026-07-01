using Documenter
using PlasticRecurrentNeuralNetworks

DocMeta.setdocmeta!(
    PlasticRecurrentNeuralNetworks,
    :DocTestSetup,
    :(using PlasticRecurrentNeuralNetworks);
    recursive=true,
)

makedocs(;
    modules=[PlasticRecurrentNeuralNetworks],
    authors="Dylan Festa <dylan.festa@gmail.com> and contributors",
    sitename="PlasticRecurrentNeuralNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://dylanfesta.github.io/PlasticRecurrentNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Rate Models" => "rate-models.md",
        "Topology Utilities" => "topology-utilities.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/PlasticRecurrentNeuralNetworks.jl",
    devbranch="main",
)
