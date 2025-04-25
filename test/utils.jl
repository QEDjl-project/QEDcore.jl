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
