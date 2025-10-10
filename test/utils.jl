"""
Returns the cartesian coordinates (E,px,py,pz) for given spherical coordiantes
(E, rho, cos_theta, phi), where rho denotes the length of the respective three-momentum,
theta is the polar- and phi the azimuthal angle.
"""
function _cartesian_coordinates(E, rho, cth, phi)
    sth = QEDcore.sq_diff_sqrt(1, cth)
    sphi, cphi = sincos(phi)
    return (E, rho * sth * cphi, rho * sth * sphi, rho * cth)
end

"""
    _is_test_platform_active(env_vars::AbstractVector{String}, default::Bool)::Bool

# Args
- `env_vars::AbstractVector{String}`: List of the names of environment variables. The value of the
    first defined variable in the list is parsed and returned.
- `default::Bool`: If none of the variables named in `env_vars` are defined, this value is returned.

# Return

Return if platform is active or not.
"""
function _is_test_platform_active(env_vars::AbstractVector{String}, default::Bool)::Bool
    for env_var in env_vars
        if haskey(ENV, env_var)
            return tryparse(Bool, ENV[env_var])
        end
    end
    return default
end
