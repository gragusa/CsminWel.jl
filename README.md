# CsminWel.jl

[![Build Status](https://travis-ci.org/gragusa/CsminWel.jl.svg?branch=master)](https://travis-ci.org/gragusa/CsminWel.jl) [![Coverage Status](https://coveralls.io/repos/gragusa/CsminWel.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/CsminWel.jl?branch=master) [![codecov.io](http://codecov.io/github/gragusa/CsminWel.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/CsminWel.jl?branch=master)

Interface to Chris Sims' `csminwel` optimization code. The code borrows from [DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl), but it is adapted to be compatible with [Optim.jl](https://github.com/JuliaOpt/Optim.jl)'s API. When the derivative of the minimand is not supplied, either Finite Difference of Forward Automatic Differentiation derivatives are used.

Differently from the solvers in `Optim.jl`, `Csminwel` returns an estimate of the inverse of the Hessian at the solution.

```julia
#=
Maximizing loglikelihood logistic models
=#
using CsminWel, StatsFuns, Random
Random.seed!(1)
x = [ones(200) randn(200,4)]
y = [rand() < 0.5 ? 1. : 0. for j in 1:200]

function loglik(beta)
    xb = x*beta
    sum(-y.*xb + log1pexp.(xb))
end

function dloglik(beta)
    xb = x*beta
    px = logistic.(xb)
    -x'*(y.-px)
end

function g!(stor, beta)
    copyto!(stor, dloglik(beta))
end

res1 = optimize(loglik, g!, zeros(5), BFGS())
res2 = optimize(loglik, g!, zeros(5), Csminwel())

## With finite-difference derivative
res3 = optimize(loglik, zeros(5), Csminwel())

## With forward AD derivative
res4 = optimize(Optim.OnceDifferentiable(loglik, zeros(5), autodiff = :forward), zeros(5), Csminwel())

## inverse Hessian
res2.invH
```
