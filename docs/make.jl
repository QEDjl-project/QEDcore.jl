using Pkg

# targeting the correct source code
# this asumes the make.jl script is located in QEDcore.jl/docs
project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")
Pkg.develop(; path=project_path)

Pkg.add(; url="https://github.com/QEDjl-project/QEDbase.jl/", rev="dev")

using QEDcore
using Documenter

# DocMeta.setdocmeta!(QEDcore, :DocTestSetup, :(using QEDcore); recursive=true)

makedocs(;
    modules=[QEDcore],
    authors="Uwe Hernandez Acosta <u.hernandez@hzdr.de>",
    sitename="QEDcore.jl",
    format=Documenter.HTML(;
        canonical="https://QEDjl-project.github.io/QEDcore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/QEDjl-project/QEDcore.jl", push_preview=false)
