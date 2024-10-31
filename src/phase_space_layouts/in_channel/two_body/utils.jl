
_order_moms(::Val{1}, ::Val{2}, P_rest, P_run) = (P_rest, P_run)
_order_moms(::Val{2}, ::Val{1}, P_rest, P_run) = (P_run, P_rest)
function _order_moms(rest_idx::Int, run_idx::Int, P_rest, P_run)
    return _order_moms(Val(rest_idx), Val(run_idx), P_rest, P_run)
end

_the_other(::Val{1}) = 2
_the_other(::Val{2}) = 1
_the_other(i::Int) = _the_other(Val(i))
