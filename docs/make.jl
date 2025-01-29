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
using DocumenterCitations
using Literate

# setup interlinks
links = InterLinks(
    "QuantumElectrodynamics" => "https://qedjl-project.github.io/QuantumElectrodynamics.jl/dev/",
    "QEDbase" => "https://qedjl-project.github.io/QEDbase.jl/dev/",
    "QEDcore" => "https://qedjl-project.github.io/QEDcore.jl/dev/",
    "QEDprocesses" => "https://qedjl-project.github.io/QEDprocesses.jl/dev/",
)

# add Bibliography
bib = CitationBibliography(joinpath(@__DIR__, "Bibliography.bib"))

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

# setup examples using Literate.jl
literate_paths = [
    Base.Filesystem.joinpath(project_path, "docs/src/tutorial/ps_def.jl"),
    Base.Filesystem.joinpath(project_path, "docs/src/tutorial/particles.jl"),
    Base.Filesystem.joinpath(project_path, "docs/src/tutorial/vectors.jl"),
]

tutorial_output_dir = joinpath(project_path, "docs/src/generated/")
!ispath(tutorial_output_dir) && mkdir(tutorial_output_dir)
@info "Literate: create temp dir at $tutorial_output_dir"

tutorial_output_dir_name = splitpath(tutorial_output_dir)[end]

pages = [
    "Home" => "index.md",
    "Tutorials" => [
        "Phase Space Definitions" => joinpath(tutorial_output_dir_name, "ps_def.md"),
        "Particles and Phase Space Points" =>
            joinpath(tutorial_output_dir_name, "particles.md"),
        "Matrix and Vector Types" => joinpath(tutorial_output_dir_name, "vectors.md"),
    ],
    "API reference" => [
        "Contents" => "library/outline.md",
        "Particles" => "library/particles.md",
        "Coordinates" => "library/coordinates.md",
        "Phase Space Definition" => "library/phasespacedef.md",
        "Phase Space Layout" => "library/phase_space_layout.md",
        "Phase Space Points" => "library/phasespacepoint.md",
        "Vector Types" => "library/vectors.md",
        "Lorentz Boosts" => "library/lorentzboosts.md",
        "Index" => "library/index.md",
    ],
    "refs.md",
]

try
    # generate markdown files with Literate.jl
    for file in literate_paths
        Literate.markdown(file, tutorial_output_dir; documenter=true)
    end
    # geneate docs with Documenter.jl

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
        plugins=[bib, links],
    )
finally
    # doing some garbage collection
    @info "GarbageCollection: remove generated landing page"
    rm(index_path)
    @info "GarbageCollection: remove generated tutorial files"
    rm(tutorial_output_dir; recursive=true)
end

deploydocs(; repo="github.com/QEDjl-project/QEDcore.jl", push_preview=false)
