using CsminWel, Test, Random

f(x) = sum(x.^2)
function grad(x)
    2*x, false
end

function g!(stor, x)
    stor[:] = 2*x
end


res0 = CsminWel.csminwel(f, grad, [0.1,0.1])
res1 = CsminWel.csminwel(f, [0.1,0.1])

res2 = CsminWel.csminwel(f, grad, [0.1,0.1])
res3 = CsminWel.csminwel(f, [0.1,0.1], H = Matrix{Float64}(I, 2, 2))


res4 = optimize(f, g!, [.1, .1], Csminwel())
res5 = optimize(f, [.1, .1], Csminwel())   ## Default to finite difference
res6 = optimize(OnceDifferentiable(f, [.1, .1]; autodiff = :forward), [.1, .1], Csminwel())


@test Optim.minimum(res0) ≈ Optim.minimum(res1) atol=1.0e-9
@test Optim.minimum(res1) ≈ Optim.minimum(res2) atol=1.0e-9
@test Optim.minimum(res2) ≈ Optim.minimum(res3) atol=1.0e-9
@test Optim.minimum(res3) ≈ Optim.minimum(res4) atol=1.0e-9
@test Optim.minimum(res4) ≈ Optim.minimum(res5) atol=1.0e-9
@test Optim.minimum(res6) ≈ Optim.minimum(res6) atol=1.0e-9

#Maximizing loglikelihood logistic models
using StatsFuns
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
res3 = optimize(loglik, zeros(5), Csminwel())
res4 = optimize(Optim.OnceDifferentiable(loglik, zeros(5), autodiff = :forward), zeros(5), Csminwel())
res5 = optimize(Optim.OnceDifferentiable(loglik, zeros(5), autodiff = :finite), zeros(5), Csminwel())


@test Optim.minimum(res1) ≈ Optim.minimum(res2) atol=1.0e-9
@test Optim.minimum(res2) ≈ Optim.minimum(res3) atol=1.0e-9
@test Optim.minimum(res3) ≈ Optim.minimum(res4) atol=1.0e-9
@test Optim.minimum(res4) ≈ Optim.minimum(res5) atol=1.0e-9


res = optimize(loglik, g!, zeros(5), Csminwel(), Optim.Options(extended_trace = true, show_trace = true, store_trace = true))

@test isa(res.trace, Array{Optim.OptimizationState{Float64,CsminWel.Csminwel},1})
