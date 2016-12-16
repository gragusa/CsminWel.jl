using CsminWel
using Base.Test

# write your own tests here
f(x) = sum(x.^2)
function grad(x)
    2*x, false
end

function g!(x, stor)
    stor[:] = 2*x
end

res1 = optimize(f, g!, [.1, .1], Csminwel())
res2 = optimize(f, [.1, .1], Csminwel())
res3 = optimize(f, [.1, .1], Csminwel(), OptimizationOptions(autodiff=true))


@test_approx_eq_eps Optim.minimum(res2)  Optim.minimum(res1) 1e-09
@test_approx_eq_eps Optim.minimum(res2)  Optim.minimum(res3) 1e-09

@test_approx_eq_eps Optim.minimizer(res2)  Optim.minimizer(res1) 1e-09
@test_approx_eq_eps Optim.minimizer(res2)  Optim.minimizer(res3) 1e-09

#=
Maximizing loglikelihood logistic models
=#
using StatsFuns
srand(1)
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

function fg!(beta, stor)
    stor[:] = dloglik(beta)
end

res1 = optimize(loglik, fg!, zeros(5), BFGS())
res2 = optimize(loglik, fg!, zeros(5), Csminwel())
res3 = optimize(loglik, zeros(5), Csminwel(), OptimizationOptions(autodiff=true))
res4 = optimize(loglik, zeros(5), Csminwel(), OptimizationOptions(autodiff=false))

@test_approx_eq_eps Optim.minimum(res2)  Optim.minimum(res1) 1e-06
@test_approx_eq_eps Optim.minimum(res2)  Optim.minimum(res3) 1e-06
@test_approx_eq_eps Optim.minimum(res4)  Optim.minimum(res3) 1e-06

@test_approx_eq_eps Optim.minimizer(res2)  Optim.minimizer(res1) 1e-06
@test_approx_eq_eps Optim.minimizer(res2)  Optim.minimizer(res3) 1e-06
@test_approx_eq_eps Optim.minimizer(res3)  Optim.minimizer(res4) 1e-06
