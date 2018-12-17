module CsminWel

using Reexport
using ForwardDiff
@reexport using Optim
using Calculus
using LinearAlgebra

import Base.minimum
import Random: AbstractRNG, MersenneTwister
import Printf.@printf
import Optim: AbstractOptimizer,
              OptimizationResults,
              MultivariateOptimizationResults,
              optimize, initial_state, minimizer,
              minimum, iterations, converged, x_converged,
              f_converged, f_tol, g_tol, g_converged, iteration_limit_reached,
              f_calls, method

#=
This code is based on a routine originally copyright Chris Sims.
See http://sims.princeton.edu/yftp/optimize/
It was transcripted in julia copyright Giuseppe Ragusa
Copyright (c) 2015, Federal Reserve Bank of New York Copyright (c) 2015, Chris Sims Copyright (c) 2015, Giuseppe Ragusa
=#

struct Csminwel <: Optim.AbstractOptimizer end

## Changed respoct to default to store Hessian matrix
mutable struct MultivariateOptimizationResultsCs{O<:AbstractOptimizer,T,N} <: OptimizationResults
    method::String
    initial_x::Array{T,N}
    minimizer::Array{T,N}
    minimum::T
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::Float64
    f_converged::Bool
    f_tol::Float64
    g_converged::Bool
    g_tol::Float64
    trace::OptimizationTrace{T,O}
    f_calls::Int
    g_calls::Int
    h_calls::Int
    invH::Array{T, 2}
end

initial_state(r::MultivariateOptimizationResultsCs) = r.initial_x
minimum(r::MultivariateOptimizationResultsCs) = r.minimum
minimizer(r::MultivariateOptimizationResultsCs) = r.minimizer
converged(r::MultivariateOptimizationResultsCs) = r.x_converged || r.f_converged || r.g_converged

x_converged(r::MultivariateOptimizationResultsCs) = r.x_converged
f_converged(r::MultivariateOptimizationResultsCs) = r.f_converged
g_converged(r::MultivariateOptimizationResultsCs) = r.g_converged

x_tol(r::MultivariateOptimizationResultsCs) = r.x_tol
f_tol(r::MultivariateOptimizationResultsCs) = r.f_tol
g_tol(r::MultivariateOptimizationResultsCs) = r.g_tol

function Base.show(io::IO, r::MultivariateOptimizationResultsCs)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(initial_state(r)[1:2], ",")
    end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(minimizer(r)[1:2], ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s\n" x_tol(r) x_converged(r)
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" f_tol(r) f_converged(r)
    @printf io "   * |g(x)| < %.1e: %s\n" g_tol(r) g_converged(r)
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Function Calls: %d\n" f_calls(r)
    return
end

const VERBOSITY = Dict(:none => 0, :low => 1, :high => 2)
const rc_messages = Dict(0 => "Standard Iteration",
1 => "zero gradient",
2 => "back and forth on step length never finished",
3 => "smallest step still improving too slow",
4 => "back and forth on step length never finished",
5 => "largest step still improving too fast",
6 => "smallest step still improving too slow, reversed gradient",
7 => "warning: possible inaccuracy in H matrix")

function Optim.optimize(f::Function,
    initial_x::Array,
    method::Csminwel,
    options::Optim.Options = Optim.Options();
    H0::Matrix             = 1e-5 .* Matrix{eltype(initial_x)}(I, length(initial_x), length(initial_x)),
    rng::AbstractRNG       = MersenneTwister(0))

    opts = extract_options(options)
    d = Optim.OnceDifferentiable(f, initial_x; autodiff = :finite)
    function gradwrap(g!, x)
        stor = Array{Float64}(undef, length(x))
        g!(stor, x)
        bad_grads = abs.(stor) .>= 1e15
        stor[bad_grads] .= 0.0
        return stor, any(bad_grads)
    end
    csminwel(f, x -> gradwrap(d.df,x), initial_x; H0 = H0, rng = rng, opts...)
end

function Optim.optimize(d::Optim.OnceDifferentiable,
    initial_x::Array,
    method::Csminwel,
    options::Optim.Options = Optim.Options();
    H0::Matrix             = 1e-5 .* Matrix{eltype(d.x_f)}(I, length(d.x_f), length(d.x_f)),
    rng::AbstractRNG       = MersenneTwister(0))

    opts = extract_options(options)

    function gradwrap(g!, x)
        stor = Array{Float64}(undef, length(x))
        g!(stor, x)
        bad_grads = abs.(stor) .>= 1e15
        stor[bad_grads] .= 0.0
        return stor, any(bad_grads)
    end
    csminwel(d.f, x -> gradwrap(d.df,x), initial_x; H0 = H0, rng = rng, opts...)
end

function extract_options(opt::Optim.Options)
    opts = Array{Any}(undef, 0)
    for name in fieldnames(typeof(opt))
        push!(opts, (name, getfield(opt, name)))
    end
    opts
end


function Optim.optimize(f::Function,
    g!::Function,
    initial_x::Array{T,1},
    method::Csminwel,
    options::Optim.Options = Optim.Options();
    H0::Matrix             = 1e-5.* Matrix{eltype(initial_x)}(I, length(initial_x), length(initial_x)),
    rng::AbstractRNG       = MersenneTwister(0)) where T <: Real

    opts = extract_options(options)

    function gradwrap(g!, x)
        stor = Array{Float64}(undef, length(x))
        g!(stor, x)
        bad_grads = abs.(stor) .>= 1e15
        stor[bad_grads] .= 0.0
        return stor, any(bad_grads)
    end
    csminwel(f, x->gradwrap(g!, x), initial_x; H0 = H0, rng = rng, opts...)
end

function csminwel(fcn::Function,
    grad::Function,
    x0::Vector;
    H0::Matrix           = 1e-5 .* Matrix{eltype(x0)}(I, length(x0), length(x0)),
    rng::AbstractRNG     = MersenneTwister(0),
    verbose::Symbol      = :high,
    x_tol::Real          = 1e-32,  # default from Optim.jl
    f_tol::Float64       = 1e-14,  # Default from csminwel
    g_tol::Real          = 1e-8,   # default from Optim.jl
    iterations::Int      = 1000,
    store_trace::Bool    = false,
    show_trace::Bool     = false,
    extended_trace::Bool = false,
    kwargs...)


    xtol, ftol, grtol  = x_tol, f_tol, g_tol

    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end

    # unpack dimensions
    nx = size(x0, 1)

    # Count function and gradient calls
    f_calls, g_calls, h_calls = 0, 0, 0

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(x0), copy(x0)

    # start with Initial Hessian
    H = H0

    # start rc parameter at 0
    rc = 0

    f_x = fcn(x0)
    f_calls += 1

    if f_x > 1e50
        throw(ArgumentError("Bad initial guess. Try again"))
    end

    gr, badg = grad(x0)
    g_calls += 1

    # Count iterations
    iteration = 0

    # Maintain a trace
    tr = OptimizationTrace{typeof(f_x), Csminwel}()
    tracing = show_trace || store_trace || extended_trace

    if tracing
        dt = Dict()
        if extended_trace
            dt["x"] = copy(x)
            dt["g(x)"] = copy(gr)
            dt["H(x)"] = copy(H)
            dt["rc"] = copy(rc)
            dt["rc_message"] = rc_messages[rc]
        end
        grnorm = norm(gr, Inf)
        Optim.update!(tr, iteration, f_x, grnorm, dt, store_trace, show_trace)
    end

    # set objects to their starting values
    retcode3 = 101

    # set up return variables so they are available outside while loop
    fh = copy(f_x)
    xh = copy(x0)
    gh = copy(x0)
    retcodeh = 1000

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence or exhaustion
    converged = false
    while !converged && iteration < iterations
        iteration += 1

        f1, x1, fc, retcode1 = csminit(fcn, x, f_x, gr, badg, H; verbose=verbose)
        f_calls += fc

        if retcode1 != 1
            if retcode1 == 2 || retcode1 == 4
                wall1, badg1 = true, true
            else
                g1, badg1 = grad(x1)
                g_calls += 1
                wall1 = badg1
            end

            # Bad gradient or back and forth on step length.  Possibly at cliff edge. Try
            # perturbing search direction if problem not 1D
            if wall1 && (length(H) > 1)

                Hcliff = H + Diagonal(diag(H)).*rand(rng, nx)

                if VERBOSITY[verbose] >= VERBOSITY[:low]
                    @printf "Cliff.  Perturbing search direction.\n"
                end

                f2, x2, fc, retcode2 = csminit(fcn, x, f_x, gr, badg, Hcliff; verbose=verbose)
                f_calls += fc

                if f2 < f_x
                    if retcode2==2 || retcode2==4
                        wall2 = true; badg2 = true
                    else
                        g2, badg2 = grad(x2)
                        g_calls += 1
                        wall2 = badg2
                        badg2
                    end

                    if wall2
                        if VERBOSITY[verbose] >= VERBOSITY[:low]
                            @printf "Cliff again.  Try traversing\n"
                        end

                        if norm(x2-x1) < 1e-13
                            f3 = f_x
                            x3 = x
                            badg3 = true
                            retcode3 = 101
                        else
                            gcliff = ( (f2-f1) / ((norm(x2-x1))^2) )*(x2-x1)
                            if (size(x0 , 2)>1)
                                gcliff = gcliff'
                            end
                            f3, x3, fc, retcode3 = csminit(fcn, x, f_x, gcliff,
                            false, Matrix{eltype(nx)}(I, nx, nx); verbose=verbose)
                            f_calls += fc

                            if retcode3==2 || retcode3==4
                                wall3 = true
                                badg3 = true
                            else
                                g3, badg3 = grad(x3)
                                g_calls += 1
                                wall3 = badg3
                            end
                        end
                    else
                        f3 = f_x
                        x3 = x
                        badg3 = true
                        retcode3 = 101
                    end
                else
                    f3 = f_x
                    x3 = x
                    badg3 = true
                    retcode3 = 101
                end
            else
                # normal iteration, no walls, or else 1D, or else we're finished here.
                f2, f3 = f_x, f_x
                badg2, badg3 = true, true
                retcode2, retcode3 = 101, 101
            end
        else
            f1, f2, f3 = f_x, f_x, f_x
            retcode2, retcode3 = retcode1, retcode1
        end

        # how to pick gh and xh
        if f3 < f_x - ftol && badg3==0
            ih = 3
            fh = f3
            xh = x3
            gh = g3
            badgh = badg3
            retcodeh = retcode3
        elseif f2 < f_x - ftol && badg2==0
            ih = 2
            fh = f2
            xh = x2
            gh = g2
            badgh = badg2
            retcodeh = retcode2
        elseif f1 < f_x - ftol && badg1==0
            ih = 1
            fh = f1
            xh = x1
            gh = g1
            badgh = badg1
            retcodeh = retcode1
        else
            fh, ih = findmin([f1 , f2 , f3])

            if ih == 1
                xh = x1
                retcodeh = retcode1
            elseif ih == 2
                xh = x2
                retcodeh = retcode2
            elseif ih == 3
                xh = x3
                retcodeh = retcode3
            end

            if @isdefined gh
                nogh = isempty(gh)
            else
                nogh = true
            end

            if nogh
                gh, badgh = grad(xh)
                g_calls += 1
            end

            badgh = true
        end

        stuck = (abs(fh-f_x) < ftol)
        if !badg && !badgh && !stuck
            H = bfgsi(H , gh-gr , xh-x; verbose=verbose)
        end

        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @printf "Improvement on iteration %d = %18.9f\n" iteration fh-f_x
        end

        if stuck
            if VERBOSITY[verbose] >= VERBOSITY[:low]
                @printf "improvement < ftol -- terminating\n"
            end
        end

        # record# retcodeh of previous x
        copyto!(x_previous, x)

        # update before next iteration
        f_x_previous, f_x = f_x, fh
        x = xh
        gr = gh
        badg = badgh

        # Check convergence
        x_converged, f_converged, gr_converged, converged =
            assess_convergence(x, x_previous, f_x, f_x_previous, gr, xtol, ftol, grtol)

        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["H(x)"] = copy(H)
                dt["rc"] = copy(rc)
                dt["rc_message"] = rc_messages[rc]
            end
            grnorm = norm(gr, Inf)
            Optim.update!(tr, iteration, f_x, grnorm, dt, store_trace, show_trace)
        end

    end

    return MultivariateOptimizationResultsCs("Csminwel", x0, x, convert(Float64, f_x),
    iteration, iteration==iterations, x_converged, xtol, f_converged, ftol,
    gr_converged, grtol, tr, f_calls, g_calls, h_calls, H)   # also return H
end

#
#=
Version of `csminwel` that will use finite differencing methods to
approximate the gradient numerically. This is convenient for cases where
you cannot supply an analytical derivative, but it is not as robust as
using the true derivative.
=#

# """
# ```
# csminwel(fcn::Function, grad::Function, x0::Vector, H0::Matrix=1e-5.*eye(length(x0)), args...;
# xtol::Real=1e-32, ftol::Float64=1e-14, grtol::Real=1e-5, iterations::Int=1000,
# store_trace::Bool = false, show_trace::Bool = false, extended_trace::Bool = false,
# verbose::Symbol = :none, rng::AbstractRNG = MersenneTwister(), kwargs...)
# ```

# Minimizes `fcn` using the csminwel algorithm.

# ### Arguments

# * `fcn::Function` : The objective function
# * `grad::Function` : The gradient of the objective function. This argument can be omitted if
# an analytical gradient is not available, which will cause a numerical gradient to be
# calculated.
# * `x0::Vector`: The starting guess for the optimizer

# ### Optional Arguments

# * `H0::Matrix`: An initial guess for the Hessian matrix -- must be
# positive definite. If none is given, then a scaled down identity
# matrix is used.
# * `args...`:  Other positional arguments to be passed to `f` on each
# function call

#     ### Keyword Arguments

#     * `ftol::{T<:Real}=1e-14`: Threshold for convergence in terms of change
#     in function value across iterations.
#         * `iterations::Int=100`: Maximum number of iterations
#         * `kwargs...`: Other keyword arguments to be passed to `f` on each
#         function call
#             """

function csminwel(fcn::Function, x0::Vector{T}; kwargs...) where T <: Real
    x0 = float(x0)
    autodiff = false
    for fn in kwargs
        if fn[1] == :autodiff
            autodiff = fn[2]
        end
    end
    grad(x::Array{T}) where T<:Number = csminwell_grad(fcn, x, Val{autodiff})
    csminwel(fcn, grad, x0; kwargs...)
end


function csminwell_grad(fcn, x, ::Type{Val{false}})
    gr = Calculus.gradient(fcn, x)
    bad_grads = abs.(gr) .>= 1e15
    gr[bad_grads] .= 0.0
    return gr, any(bad_grads)
end

function csminwell_grad(fcn, x, ::Type{Val{true}})
    gr = ForwardDiff.gradient(fcn, x)
    bad_grads = abs.(gr) .>= 1e15
    gr[bad_grads] .= 0.0
    return gr, any(bad_grads)
end



function csminit(fcn, x0::Vector{T}, f0, g0, badg, H0, args...; verbose::Symbol=:high, kwargs...) where T <: Real
    x0 = float(x0)
    angle = .005

    #(0<THETA<.5) THETA near .5 makes long line searches, possibly fewer iterations.
    theta = .3
    fchange = 1000
    minlamb = 1e-9
    mindfac = .01
    f_calls = 0
    lambda = 1.0
    xhat = x0
    f = f0
    fhat = f0
    gr = g0
    gnorm = norm(gr)

    if gnorm < 1e-12 && !badg
        # gradient convergence
        retcode = 1
        dxnorm = 0.0
    else
        # with badg true, we don't try to match rate of improvement to directional
        # derivative.  We're satisfied just to get some improvement in f.
        dx = vec(-H0*gr)
        dxnorm = norm(dx)

        if dxnorm > 1e12

            if VERBOSITY[verbose] >= VERBOSITY[:low]
                @printf "Near singular H problem.\n"
            end

            dx = dx * fchange / dxnorm
        end

        dfhat = dot(dx, g0)

        if !badg
            # test for alignment of dx with gradient and fix if necessary
            a = -dfhat / (gnorm*dxnorm)

            if a < angle
                dx -= (angle*dxnorm/gnorm + dfhat/(gnorm*gnorm)) * gr
                dx *= dxnorm/norm(dx)
                dfhat = dot(dx, gr)

                if VERBOSITY[verbose] >= VERBOSITY[:low]
                    @printf "Correct for low angle %f\n" a
                end
            end
        end

        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @printf "Predicted Improvement: %18.9f\n" (-dfhat/2)
        end
        # Have OK dx, now adjust length of step (lambda) until min and max improvement rate
        # criteria are met.
        done = false
        fact = 3.0
        shrink = true
        lambda_min = 0.0
        lambda_max = Inf
        lambda_peak = 0.0
        f_peak = f0
        lambda_hat = 0.0

        while !done
            if size(x0, 2) > 1
                dxtest = x0 + dx' * lambda
            else
                dxtest = x0 + dx * lambda
            end

            f = fcn(dxtest, args...; kwargs...)

            if VERBOSITY[verbose] >= VERBOSITY[:high]
                @printf "lambda = %10.5f; f = %20.7f\n" lambda f
            end

            if f < fhat
                fhat = f
                xhat = dxtest
                lambdahat = lambda
            end

            f_calls += 1
            shrink_signal = (!badg & (f0-f < maximum([-theta*dfhat*lambda 0]))) || (badg & ((f0-f) < 0))

            grow_signal = !badg && (lambda > 0)  &&
            (f0-f > -(1-theta)*dfhat*lambda)

            if shrink_signal && ((lambda > lambda_peak) || lambda < 0 )
                if (lambda > 0) && ((!shrink) || (lambda/fact <= lambda_peak))
                    shrink = true
                    fact = fact^.6
                    while lambda / fact <= lambda_peak
                        fact = fact^.6
                    end

                    if abs(fact - 1.0) < mindfac
                        if abs(lambda) < 4
                            retcode = 2
                        else
                            retcode = 7
                        end

                        done = true
                    end
                end

                if lambda < lambda_max && lambda > lambda_peak
                    lambda_max = lambda
                end

                lambda /= fact
                if abs(lambda) < minlamb
                    if (lambda > 0) && (f0 <= fhat)
                        lambda = -lambda*fact^6
                    else
                        if lambda < 0
                            retcode = 6
                        else
                            retcode = 3
                        end
                        done = true
                    end
                end


            elseif (grow_signal && lambda > 0) || (shrink_signal &&
                ((lambda <= lambda_peak) && (lambda > 0)))
                if shrink
                    shrink = false
                    fact = fact^.6
                    if abs(fact - 1) < mindfac
                        if abs(lambda) < 4
                            retcode = 4
                        else
                            retcode = 7
                        end
                        done = true
                    end
                end

                if f < f_peak && lambda > 0
                    f_peak = f
                    lambda_peak = lambda
                    if lambda_max <= lambda_peak
                        lambda_max = lambda_peak * fact^2
                    end
                end

                lambda *= fact
                if abs(lambda) > 1e20
                    retcode = 5
                    done = true
                end
            else
                done = true
                if fact < 1.2
                    retcode = 7
                else
                    retcode = 0
                end
            end
        end
    end

    if VERBOSITY[verbose] >= VERBOSITY[:high]
        @printf "Norm of dx %10.5f\n" dxnorm
    end

    return fhat, xhat, f_calls, retcode
end

"""
```
bfgsi(H0, dg, dx)
```
### Arguments
- `H0`: hessian matrix
- `dg`: previous change in gradient
- `dx`: previous change in x
"""
function bfgsi(H0, dg, dx; verbose::Symbol = :high)
    if size(dg, 2) > 1
        dg = dg'
    end

    if size(dx, 2) > 1
        dx = dx'
    end

    Hdg = H0*dg
    dgdx = dot(dx, dg)

    H = H0
    if abs(dgdx) > 1e-12
        H += (dgdx.+(dg'*Hdg)).*(dx*dx')/(dgdx^2) - (Hdg*dx'.+dx*Hdg')/dgdx
    elseif norm(dg) < 1e-7
        # gradient is super small so don't worry updating right now
        # do nothing
    else
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            @warn("bfgs update failed")
            @printf "|dg| = %f, |dx| = %f\n" (norm(dg)) (norm(dx))
            @printf "dg'dx = %f\n" dgdx
            @printf "|H*dg| = %f\n" (norm(Hdg))
        end
    end

    return H
end

function assess_convergence(x::Array,
    x_previous::Array,
    f_x::Real,
    f_x_previous::Real,
    gr::Array,
    xtol::Real,
    ftol::Real,
    grtol::Real)
    x_converged, f_converged, gr_converged = false, false, false

    if Optim.maxdiff(x, x_previous) < xtol
        x_converged = true
    end

    # Relative Tolerance
    # if abs(f_x - f_x_previous) / (abs(f_x) + ftol) < ftol || nextfloat(f_x) >= f_x_previous
    # Absolute Tolerance
    if abs(f_x - f_x_previous) < ftol
        f_converged = true
    end

    if norm(vec(gr), Inf) < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return x_converged, f_converged, gr_converged, converged
end

## CSMINWEL
function getgradient(fcn, grad_f, s::Csminwel, ::Type{Val{true}}, ::Type{Val{false}})
    function gradient(x)
        gr = similar(x)
        grad_f(gr, x)
        bad_grads = abs.(gr) .>= 1e15
        gr[bad_grads] .= 0.0
        return gr, any(bad_grads)
    end
end

function getgradient(fcn, grad_f, s::Csminwel, ::Type{Val{false}}, ::Type{Val{true}})
    function gradient(x)
        gr = similar(x)
        ForwardDiff.gradient!(gr, fcn, x)
        bad_grads = abs.(gr) .>= 1e15
        gr[bad_grads] .= 0.0
        return gr, any(bad_grads)
    end
end

function getgradient(fcn, grad_f, s::Csminwel, ::Type{Val{false}}, ::Type{Val{false}})
    function gradient(x)
        gr = Calculus.gradient(fcn, x)
        bad_grads = abs.(gr) .>= 1e15
        gr[bad_grads] .= 0.0
        return gr, any(bad_grads)
    end
end

export Csminwel

end
