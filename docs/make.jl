using Pkg

# targeting the correct source code
# this asumes the make.jl script is located in QEDcore.jl/docs
project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")
Pkg.develop(; path=project_path)

using QEDbase
using QEDcore
using QEDprocesses

using Documenter
using DocumenterInterLinks

# setup interlinks
links = InterLinks(
    "QuantumElectrodynamics" => "https://qedjl-project.github.io/QuantumElectrodynamics.jl/dev/",
    "QEDbase" => "https://qedjl-project.github.io/QEDbase.jl/dev/",
    "QEDcore" => "https://qedjl-project.github.io/QEDcore.jl/dev/",
    "QEDprocesses" => "https://qedjl-project.github.io/QEDprocesses.jl/dev/",
)

# some paths for links
readme_path = joinpath(project_path, "README.md")
index_path = joinpath(project_path, "docs/src/index.md")
license_path = "https://github.com/QEDjl-project/QEDcore.jl/blob/main/LICENSE"

# Copy README.md from the project base folder and use it as the start page
open(readme_path, "r") do readme_in
    readme_string = read(readme_in, String)

    # replace relative links in the README.md
    readme_string = replace(readme_string, "[MIT](LICENSE)" => "[MIT]($(license_path))")

    open(index_path, "w") do readme_out
        write(readme_out, readme_string)
    end
end

pages = [
    "Home" => "index.md",
    "API reference" => [
        "Contents" => "library/outline.md",
        "Particles" => "library/particles.md",
        "Phase Space Definition" => "library/phasespacedef.md",
        "Phase Space Points" => "library/phasespacepoint.md",
        "Vector Types" => "library/vectors.md",
    ],
]

try
    makedocs(;
        modules=[QEDcore],
        checkdocs=:exports,
        authors="Uwe Hernandez Acosta",
        repo=Documenter.Remotes.GitHub("QEDjl-project", "QEDcore.jl"),
        sitename="QEDcore.jl",
        format=Documenter.HTML(;
            prettyurls=get(ENV, "CI", "false") == "true",
            canonical="https://qedjl-project.gitlab.io/QEDcore.jl",
            assets=String[],
        ),
        pages=pages,
    )
finally
    @info "Garbage collection: remove landing page"
    rm(index_path)
end

deploydocs(; repo="github.com/QEDjl-project/QEDcore.jl", push_preview=false)
