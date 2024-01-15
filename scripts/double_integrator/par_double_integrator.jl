using DrWatson
@quickactivate "nn_controller"
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqUncertainty #, DifferentialEquations.EnsembleAnalysis #DiffEqFlux
# use SciMLSensitivity instead of DiffEqSensitivity
using Optimization, OptimizationOptimisers, OptimizationNLopt, OptimizationOptimJL
using Flux
using Distributions
using Plots
using ForwardDiff, NLopt
using LinearAlgebra
using IntegralsCuba, IntegralsCubature
using Sobol
using ModelingToolkit

gr()
# include("src/Adam.jl")
# plotlyjs(legend=:outertopright,guidefont = 14, legendfontsize = 14,tickfont = 10,fg_legend = :transparent)

#+ Define controller to use
Width = 12
act = relu # tanh relu leakyrelu
clmp = hardtanh #hardtanh tanh
controller = f64(Flux.Chain(
    Flux.Dense(3, Width, act),
    Flux.Dense(Width, Width, act),
    Flux.Dense(Width, 1))) # Note had to make it to accept float 64 iwth f64
p_nn, re = Flux.destructure(controller) # destructure  into a vector of parameters
n_weights = length(p_nn)
const x_scaling = [18; 8]
const origin = ([0; 0] ./ x_scaling .+ 0.5)

function nn(x, θ)
    xx = (x[1:2] ./ x_scaling .+ 0.5)
    _p = x[3] - 3.0
    u = u_norm * (clmp(re([xx; _p], θ)[1] - re([origin; _p], θ)[1])) # clmp = hardtanh = mid
    return u
end
println("Number of weights: $n_weights")

const u_norm = 3.e0
const P = [2.177565296080891 1.2571126613074837; 1.2571126613074837 1.480332243219584]

const Ad = Float64[1.0 1.0; 0.0 1.0]
const Bd = Float64[0.0; 1.0]
function disc_system!(xnp1, xn, θ, t)
    xview = [@view xn[1:2]; xn[5]]
    u = nn(xview, θ)

    xnp1[1] = xn[1] + xn[2]
    xnp1[2] = xn[2] + u
    xnp1[3] = xn[3] + (xn[1] * xn[1] + 0.05 * xn[2] * xn[2] + 0.1 * u * u)
    xnp1[4] = xn[4] + max(0.0, xn[2] - xn[5])^2 + max(0.0, -xn[2] - xn[5])^2
    xnp1[5] = xn[5] #	
    nothing
end

system!(xnp1, xn, θ, t) = disc_system!(xnp1, xn, θ, t)
#+ Problem set up
const x0_dist = [Uniform(-9.e0, 9.e0), Uniform(-3.e0, 3.e0)] #[Uniform(-1., 1.);Uniform(-1., 1.)]
const con_dist = Uniform(3.0, 4.0)


x0 = Float64[rand.(x0_dist); 0.e0; 0.e0; rand.(con_dist)]
const x_dim = 2
#+ timepoints

const t0 = 0.0e0
const tf = 4.e0
const tspan = (t0, tf)
# tsteps = t0:Δt:tf
# tlength=length(tsteps)

#+ check set-up
prob = DiscreteProblem(disc_system!, x0, tspan, p_nn)
prob_exp = DiscreteProblem(disc_system!, x0, tspan, p_nn, save_everystep=false)

const _tspan = (t0, tf)
const _prob_exp = ODEProblem(system!, x0, _tspan, p_nn, save_everystep=false) # still solved with FunctionMap so a discrete problem.

println("Checking ODE setup")
#ode_alg = AutoTsit5(Rosenbrock23())
ode_alg = FunctionMap()
#@time sol = solve(prob, ode_alg,save_everystep=false)
@time sol = solve(prob, ode_alg, save_everystep=false)

# create sobol samples from x0_dist
const lb = [x0_dist[i].a for i in 1:x_dim];
const ub = [x0_dist[i].b for i in 1:x_dim];
push!(lb, con_dist.a);
push!(ub, con_dist.b);
sobol_samples = SobolSeq(lb, ub)

n_s = 100;
skip(sobol_samples, 100);
x0_samples = reduce(hcat, next!(sobol_samples) for i = 1:n_s)'
x0_samples[1,:] = lb;
x0_samples[2,:] = ub;
function cb(p_nn, l)
    println("Current loss is: $l")
    false
end
function rand_cb(p_nn, l)
    println("Current loss is: $(printE_loss(p_nn))")
    false
end
function cb_iter(p_nn, l, k)
    println("Iteration $k, loss: $l")
    false
end

function obs_loss(sol, ρ, λ)
    # running and terminal cost
    h = sol[4, end] + max(0.0, sol[2, end] - sol[5, 1])^2 + max(0.0, -sol[2, end] - sol[5, 1])^2
    return sol[3, end] + sol[1:2, end]' * P * sol[1:2, end] + 0.5 * ρ * h .^ 2 - λ * h
end
function obs_con(sol)
    return sol[4, end] + max(0.0, sol[2, end] - sol[5, 1])^2 + max(0.0, -sol[2, end] - sol[5, 1])^2
end
function E_con(p_nn, prob, tol=1e-3)
    e = expectation(obs_con, prob, [x0_dist; 0.e0; 0.e0; con_dist], p_nn, Koopman(), ode_alg;
        quadalg=IntegralsCubature.HCubatureJL(), iabstol=tol, ireltol=tol)[1]
    return e
end
const tol = 1e-2
#IntegralsCuba.CubaSUAVE() # IntegralsCubature.HCubatureJL() # IntegralsCuba.CubaCuhre() IntegralsCubature.CubatureJLp
function E_loss(p_nn, param, obs_loss; batch=0)
    e = expectation(obs_loss, param[1], [x0_dist; 0.e0; 0.e0; con_dist], p_nn, Koopman(), ode_alg;
        quadalg=IntegralsCubature.HCubatureJL(), iabstol=param[2], ireltol=param[2],
		batch=batch,
		seed=rand(Int))[1]
    return e
end
# E_loss(p_nn) = E_loss(p_nn, prob_exp)
# output_func(sol,i) = (sol[3,end]+ sol[1:2,end]'*P*sol[1:2,end],false)
output_func(sol, i, ρ, λ) = (sol[3, end] + sol[1:2, end]' * P * sol[1:2, end] + 0.5 * ρ * sol[4, end] .^ 2 - λ * sol[4, end], false)

function rand_loss(p_nn, _output_func)
    # x0_samples[rand(1:n_s),:] next!(s)
    prob_func(prob, i, repeat) = remake(_prob_exp, p=p_nn, u0=[x0_samples[rand(1:n_s), 1:2]; 0.e0; 0.e0; x0_samples[rand(1:n_s), 3]], tspan=_tspan)
    # prob_func(prob,i,repeat) = remake(_prob_exp, p=p_nn, u0 = [next!(sobol_samples); 0.e0; 0.e0],tspan=_tspan) 
    ensemble_prob = EnsembleProblem(prob_exp, prob_func=prob_func,
        output_func=_output_func,)
    sim = solve(ensemble_prob, ode_alg, EnsembleThreads(), trajectories=1)
    return mean(sim)

end

function run_opt(solver, opt_f, max_iters, ini_guess, _prob_exp)
    opt_prob = OptimizationProblem(opt_f, ini_guess, _prob_exp)
    opt_sol = solve(
        opt_prob,
        solver,
        maxiters=max_iters,
        progress=true,
        callback=cb,
        #abstol=tol,reltol=tol
    )
    return opt_sol
end


# options: AutoReverseDiff(compile=false), AutoForwardDiff(), 
function aug_Lag_random(loss_func, prob_exp, ini_guess, solver; k_max::Int=50, λ::Real=0.0, ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
    _output_func(sol, i) = output_func(sol, i, ρ, λ)
    opt_rand = OptimizationFunction(loss_func, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_rand, p_nn, _output_func)
    h = -1.0

    for k in 1:k_max
        println("Iteration: $k")
        _output_func(sol, i) = output_func(sol, i, ρ, λ)
        _opt_problem = remake(opt_prob, u0=ini_guess, p=_output_func)
        opt_sol = solve(_opt_problem, solver, maxiters=inner_max_iter, progress=true, callback=throttle_rand, save_best=false)
        h = E_con(opt_sol.u, prob_exp)
        if h[1] < con_tol
            println("Constraint satisfied.")
            return opt_sol, h, ρ, λ
        else
            println("Constraint violation: $(h[1])")
            ini_guess .= opt_sol.u
        end
        if k == k_max
            println("Maximum number of iterations reached.")
            return opt_sol, h, ρ, λ
        end
        ρ *= γ
        λ -= ρ * h[1]
    end
    return opt_sol, h, ρ, λ  # should never reach here
end

printing_loss(sol) = obs_loss(sol, 0.0, 0.0)
printE_loss(x) = E_loss(x, (_prob_exp, 1e-1), printing_loss)


println("Loss before optimization: $(printE_loss(p_nn))")
throttle_cb = Flux.throttle(cb, 10)
throttle_rand = Flux.throttle(rand_cb, 20)

println("Starting optimization with AMSGrad")
opt_sol, h, ρ, λ = aug_Lag_random(rand_loss, _prob_exp, p_nn, Optimisers.AMSGrad(); k_max=50, inner_max_iter=6000, con_tol=1e-3, ρ=16)

println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")
println("Plotting policy")

levels = range(-u_norm, u_norm, step=0.2)
x1s = range(lb[1], ub[1], length=200)
x2s = range(lb[2], ub[2], length=200)
#k1(x1,x2) = u_norm*clmp.(re([x1,x2],opt_sol.u) -re(Float64[0.0, 0.0], opt_sol.u))[1]
k1(x1, x2) = nn([x1, x2, 3.0], opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis, levels=levels)
savefig(cont_plot, plotsdir("par_double_int", "nn_policy_AMSgrad.svg"))

NN_params = opt_sol.u
save_params = @strdict NN_params controller u_norm clmp ρ λ

println("Saving NN parameters")

file_name = datadir("par_double_int/disc_nn__intermed") * ".jld2"
save(file_name, save_params)

#test E_loss(opt_sol.u, _prob_exp)
# E_loss(x) = E_loss(x, _prob_exp)
# ForwardDiff.gradient(E_loss, opt_sol.u)

# using Zygote
# Zygote.gradient(E_loss, opt_sol.u)
# options: AutoReverseDiff(compile=false), AutoForwardDiff(), 

function aug_Lag_E(E_loss, obs_loss, ini_guess, solver, solver_param, prob_exp; k_max::Int=50, λ::Real=0.0, ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
    p0 = copy(ini_guess)
    _obs(sol) = obs_loss(sol, ρ, λ)
    _E_loss(x, p) = E_loss(x, p, _obs)

    opt_f = OptimizationFunction(_E_loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_f, p0, solver_param)

    h = -1.0

    for k in 1:k_max
        println("Iteration: $k")
        _obs(sol) = obs_loss(sol, ρ, λ)
        _E_loss(x, p) = E_loss(x, p, _obs)
        _opt_problem = remake(opt_prob, u0=p0)
        if solver == NLopt.LD_LBFGS() || solver == NLopt.LD_MMA()
            opt_sol = solve(_opt_problem, solver, maxiters=inner_max_iter, progress=true, callback=throttle_cb)
        else
            opt_sol = solve(_opt_problem, solver, maxiters=inner_max_iter, progress=true, callback=throttle_rand, save_best=false)
        end
        h = E_con(opt_sol.u, prob_exp, con_tol)
        if h[1] < con_tol
            println("Constraint satisfied.")
            return opt_sol, h, ρ, λ
        else
            println("Constraint violation: $(h[1])")
            p0 .= opt_sol.u
        end
        if k == k_max
            println("Maximum number of iterations reached.")
            return opt_sol, h, ρ, λ
        end
        ρ *= γ
        λ -= ρ * h[1]
    end
    return opt_sol, h, ρ, λ  # should never reach here
end

println("Starting optimization with expectation eval.")

const _param = (_prob_exp, 5e-3);

param_vec = opt_sol.u;

opt_sol, h, _ρ, _λ = aug_Lag_E(E_loss, obs_loss, param_vec, Optimisers.AMSGrad(1.0f-4), _param, _prob_exp; λ=λ, ρ=ρ, inner_max_iter=1000, con_tol=1e-4, k_max=10)

println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")
ρ, λ = _ρ, _λ

NN_params = opt_sol.u;
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller u_norm clmp ρ λ
file_name = datadir("par_double_int/disc_nn_ams") * ".jld2"
save(file_name, save_params)


# finish with LD_LBFGS
const _param = (_prob_exp, 1e-3);
param_vec = opt_sol.u;

opt_sol, h, _ρ, _λ = aug_Lag_E(E_loss, obs_loss, param_vec, NLopt.LD_LBFGS(), _param, _prob_exp; λ=λ, ρ=ρ, inner_max_iter=20, k_max=10, con_tol=1e-4)

# NLopt.LD_MMA NLopt.LD_LBFGS NLopt.LD_SLSQP Optimisers.AMSGrad()
println("Loss after LBFGS: $(printE_loss(opt_sol.u))")
obj = printE_loss(opt_sol.u);
NN_params = opt_sol.u;
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller u_norm clmp ρ λ;

println("Saving NN parameters")

file_name = datadir("par_double_int/disc_nn_hrdtanh_4") * ".jld2"

save(file_name, save_params);


#=
loaded= load(file_name)
NN_params = copy(loaded["NN_params"])
ρ = copy(loaded["ρ"])
λ = copy(loaded["λ"])
param_vec = copy(NN_params)
=#
#=
println("Plotting policy")

k1(x1, x2) = nn([x1, x2, 3.], opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis, levels=levels)

# save figures
savefig(cont_plot, plotsdir("disc_double_int_nn_policy.png"))
=#