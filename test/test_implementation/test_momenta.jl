
@inline function QEDbase._build_momenta(
    ::TestProcess, ::TestModel, ::TestInPhaseSpaceLayout, in_coords
)
    return _groundtruth_in_moms(in_coords)
end

@inline function QEDbase._build_momenta(
    proc::TestProcess,
    model::TestModel,
    in_moms::NTuple{NIN,AbstractFourMomentum},
    out_psl::TestOutPhaseSpaceLayout,
    out_coords,
) where {NIN}
    return _groundtruth_out_moms(in_moms, out_coords)
end
