using DrWatson 
@quickactivate "nn_controller"
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqUncertainty 
using Optimization, OptimizationOptimisers, OptimizationNLopt, OptimizationOptimJL
using Flux
using Distributions
using Plots, LaTeXStrings
using ForwardDiff, NLopt
using LinearAlgebra
using IntegralsCuba, IntegralsCubature
using Sobol
using SparseArrays

gr()
# plotlyjs(legend=:outertopright,guidefont = 14, legendfontsize = 14,tickfont = 10,fg_legend = :transparent)
include(scriptsdir("6_LTI","par.jl"))

#+ Define controller to use
# hardtanh relu sigmoid 
Width = 10
act = tanh #leakyrelu
clmp = tanh # hardtanh
controller = f64(Flux.Chain(
Flux.Dense(nx, Width, act),
Flux.Dense(Width, Width, act),
Flux.Dense(Width, Width, act),
Flux.Dense(Width, nu ))) 
p_nn,re = Flux.destructure(controller) # destructure  into a vector of parameters
n_weights = length(p_nn)

const origin = zeros(6)

function nn(x,θ)
	return clmp.(re(x,θ) - re(origin,θ))
end


function _bd_penal(x)
	return 10*(sum( (max.(0.,x_lbd .- x)).^2 ) + sum((max.(0., x .- x_upd)).^2))
end


function disc_system!(xnp1,xn,θ,t)
	xview = @view xn[1:nx]
	u = nn(xview,θ)

	xnp1[1:nx] = Ad * xview + Bd * u

	xnp1[obj_indx] = xn[obj_indx] + xview'*Q*xview + u'*R*u 
	xnp1[con_indx] = xn[con_indx] + _bd_penal(xview)
	nothing
end



system!(xnp1,xn,θ,t) = disc_system!(xnp1,xn,θ,t) 
#+ Problem set up
const x0_dist =  Uniform(-0.44e0, 0.44e0).*ones(nx) 
_xx0 = ones(nx)*0.44
x0 = Float64[_xx0; 0.e0; 0.e0]
#+ timepoints

const t0 = 0.e0
const tf = 8.e0 
const tspan = (t0, tf)
# tsteps = t0:Δt:tf
# tlength=length(tsteps)

#+ check set-up
prob = DiscreteProblem(disc_system!, x0, tspan, p_nn)
prob_exp = DiscreteProblem(disc_system!, x0, tspan, p_nn,save_everystep=false)	 

println("Checking setup")

ode_alg = FunctionMap()
@time sol = solve(prob, ode_alg, save_everystep=false)

# create sobol samples from x0_dist
const lb = [x0_dist[i].a for i in 1:nx]
const ub = [x0_dist[i].b for i in 1:nx]

sobol_samples = SobolSeq(lb, ub)

n_s = 500
skip(sobol_samples, n_s) 
const x0_samples = reduce(hcat, next!(sobol_samples) for i = 1:n_s)'
x0_samples[1,:] .= lb
x0_samples[2,:] .= ub
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
	x = @view sol[1:nx,end]
	# running and terminal cost
	h = sol[con_indx,end] + _bd_penal(x)
	return sol[obj_indx,end] + x'*P*x + 0.5*ρ*h .^2 - λ*h
end
function obs_con(sol)
	x = @view sol[1:nx,end]
	return sol[con_indx,end]+ _bd_penal(x)
end
function E_con(p_nn,prob, tol=1e-2)
	e = expectation(obs_con, prob, [x0_dist; 0.e0; 0.e0], p_nn, Koopman(), ode_alg;
	quadalg = quad_alg, iabstol=tol,ireltol=tol)[1]
	return e
end
const tol = 1e-2

function E_loss(p_nn, param, obs_loss)
	e = expectation(obs_loss, param[1], [x0_dist; 0.e0; 0.e0], p_nn, Koopman(), ode_alg;
	quadalg = quad_alg, iabstol=param[2],ireltol=param[2], seed = rand(Int))[1]
	return e 
end

function output_func(sol,i, ρ, λ) 
	x = @view sol[1:nx,end]
	return (sol[obj_indx,end]+ x'*P*x + 0.5*ρ*sol[con_indx,end] .^2 - λ*sol[con_indx,end], false)
end

function rand_loss(p_nn, _output_func)

	prob_func(prob,i,repeat) = remake(_prob_exp, p=p_nn, u0 = [x0_samples[rand(1:n_s), :]; 0.e0; 0.e0],tspan=_tspan) 

	ensemble_prob = EnsembleProblem(prob_exp, prob_func=prob_func, 
	output_func=_output_func,)
	sim = solve(ensemble_prob, ode_alg, EnsembleThreads(),trajectories=1)
	return mean(sim) 

end

function run_opt(solver, opt_f, max_iters, ini_guess, _prob_exp)
	opt_prob = OptimizationProblem(opt_f, ini_guess, _prob_exp)
	opt_sol = solve(opt_prob, solver, maxiters = max_iters, progress = true, callback = cb,
	)
	return opt_sol
end

const _tspan=(t0,tf)
const _prob_exp = ODEProblem(system!, x0, _tspan, p_nn, save_everystep=false) #* "ODEProblem" but will be solved by FunctionMap i.e. discrete time.	

function aug_Lag_random(loss_func, prob_exp, ini_guess, solver; k_max::Int= 3,	λ::Real = 0., ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
	_output_func(sol,i) = output_func(sol,i, ρ, λ)
	opt_rand = OptimizationFunction(loss_func, Optimization.AutoForwardDiff())
	opt_prob = OptimizationProblem(opt_rand, p_nn, _output_func)
	h=-1.

	for k in 1 : k_max
		println("Iteration: $k")
		_output_func_(sol,i) = output_func(sol,i, ρ, λ)
		_opt_problem = remake(opt_prob,u0=ini_guess, p =_output_func_)
		opt_sol = solve(_opt_problem, solver, maxiters = inner_max_iter, progress = true, callback = throttle_rand,save_best=false)
		h = E_con(opt_sol.u, prob_exp,con_tol)
		if h[1] < con_tol
			println("Constraint satisfied.")
			return opt_sol,h,ρ,λ
		else
			println("Constraint violation: $(h[1])")
			ini_guess .= opt_sol.u
		end
		if k==k_max
			println("Maximum number of iterations reached.")
			return opt_sol,h,ρ,λ
		end
		ρ *= γ
		λ -= ρ*h[1]
	end
	return opt_sol,h,ρ,λ  # should never reach here
end

printing_loss(sol) = obs_loss(sol, 0., 0.);
printE_loss(x) = E_loss(x,  (_prob_exp, 1e-2),printing_loss);
quad_alg = IntegralsCuba.CubaSUAVE(); # IntegralsCubature.HCubatureJL() IntegralsCuba.CubaSUAVE()  IntegralsCuba.CubaCuhre()  
# see https://docs.sciml.ai/Integrals/stable/solvers/IntegralSolvers/
println("Loss before optimization: $(printE_loss(p_nn))")

throttle_cb = Flux.throttle(cb,10)
throttle_rand = Flux.throttle(rand_cb,10)

println("Starting optimization with AMSGrad")
time_elapsed = @elapsed opt_sol,h,ρ,λ  = aug_Lag_random(rand_loss, _prob_exp, p_nn, Optimisers.AMSGrad(); k_max= 50,inner_max_iter=10000, con_tol = 1e-1, ρ=1024.)


println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")

NN_params = opt_sol.u
save_params = @strdict NN_params controller clmp

println("Saving NN parameters")

file_name = datadir("par_6LTI/nn__intermed")*".jld2"
save(file_name, save_params) 

function aug_Lag_E(E_loss, obs_loss, ini_guess, solver, solver_param,prob_exp; k_max::Int= 3,	λ::Real = 0., ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
	p0 = copy(ini_guess)
	_obs(sol) = obs_loss(sol, ρ, λ)
	_E_loss(x,p) = E_loss(x,p, _obs)

	opt_f = OptimizationFunction(_E_loss, Optimization.AutoForwardDiff())
	opt_prob = OptimizationProblem(opt_f, p0, solver_param)

	h=-1.

	for k in 1 : k_max
		println("Iteration: $k")
		_obs(sol) = obs_loss(sol, ρ, λ)
		_E_loss(x,p) = E_loss(x,p, _obs)
		_opt_problem = remake(opt_prob,u0=p0)
		if solver ==  NLopt.LD_LBFGS() || solver == NLopt.LD_MMA()
			opt_sol = solve(_opt_problem, solver, maxiters = inner_max_iter, progress = true, callback = throttle_cb)
		else
			opt_sol = solve(_opt_problem, solver, maxiters = inner_max_iter, progress = true, callback = throttle_rand,save_best=false)
		end
		h = E_con(opt_sol.u, prob_exp,con_tol)
		if h[1] < con_tol
			println("Constraint satisfied.")
			return opt_sol,h,ρ,λ
		else
			println("Constraint violation: $(h[1])")
			p0 .= opt_sol.u
		end
		if k==k_max
			println("Maximum number of iterations reached.")
			return opt_sol,h,ρ,λ
		end
		ρ *= γ
		λ -= ρ*h[1]
	end
	return opt_sol,h,ρ,λ  # should never reach here
end

println("Starting optimization with expectation eval.")

const _param = (_prob_exp, 5e-1);

@time E_con(opt_sol.u, _prob_exp,1e-1)

param_vec = opt_sol.u
aug_Lag_E(E_loss, obs_loss, param_vec, Optimisers.AMSGrad(1f-4), _param,_prob_exp; λ=λ, ρ=ρ, inner_max_iter=1, k_max=1); # just to compile

temp = @elapsed opt_sol,h,_ρ,_λ  = aug_Lag_E(E_loss, obs_loss, param_vec, Optimisers.AMSGrad(1f-4), _param,_prob_exp; λ=λ, ρ=ρ, inner_max_iter=500, con_tol=1e-1, k_max=50)

time_elapsed += temp

@time println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")
ρ,λ =_ρ,_λ

NN_params = opt_sol.u
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller clmp ρ λ 

println("Saving NN parameters")

file_name = datadir("par_6LTI/nn_ams")*".jld2"
save(file_name, save_params) 
# finish with LD_LBFGS
const _param = (_prob_exp, 1e-1);
quad_alg = IntegralsCuba.CubaSUAVE(); #* for the last section, to make certain that the integral is accurate enough

temp_vec = copy(opt_sol.u)

param_vec = opt_sol.u;
temp = @elapsed opt_sol,h,_ρ,_λ  = aug_Lag_E(E_loss, obs_loss, param_vec, NLopt.LD_LBFGS(), _param,_prob_exp; λ=λ, ρ=ρ, inner_max_iter=20, con_tol=1e-1, k_max=10)

time_elapsed += temp

# NLopt.LD_MMA NLopt.LD_LBFGS NLopt.LD_SLSQP Optimisers.AMSGrad()
println("Loss after LBFGS: $(printE_loss(opt_sol.u))")
obj  = printE_loss(opt_sol.u)
NN_params = opt_sol.u
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller clmp ρ λ

println("Saving NN parameters")

file_name = datadir("par_6LTI/nn")*".jld2";
save(file_name, save_params) 

loaded=load(file_name);
NN_params = copy(loaded["NN_params"]);
p_nn = copy(NN_params);
controller = loaded["controller"]
clmp = loaded["clmp"]

_,re = Flux.destructure(controller); # destructure  into a vector of parameters

using BenchmarkTools
@benchmark nn(x0[1:6],NN_params)

label = Matrix(undef,1,6)
[label[1,i] = L"x_%$i" for i in 1:nx]

line_style = @dict w =3  ylims=(-1.2, 1.2) linetype=:steppost label = label xlabel=L"t" ylabel=L"x" labelfontsize=20 tickfontsize=10 legendfontsize=14 legend=:outertopright 

#+ check set-up
x0 = [ones(nx)*0.44; 0.e0; 0.e0]
test_prob = DiscreteProblem(disc_system!, x0, tspan, NN_params)
@time test_sol = solve(test_prob, ode_alg, save_everystep=true)

t_point = test_sol.t
x_policy = Array(test_sol)[1:6,:];



x_array = Array(test_sol)[1:6,:];
U_array = zeros(2,length(test_sol.t));
for i=1:length(test_sol.t)
	U_array[:,i] = nn(x_array[:,i],NN_params)
end

plot(U_array')

mpc_X = [0.4400   -0.0225   -0.2596   -0.0624   -0.0150    0.0009   -0.0029    0.0008   -0.0006
    0.4400    0.9871   -0.1727   -0.0218   -0.0310    0.0111   -0.0065    0.0028   -0.0015
    0.4400    0.4823    0.0292   -0.0321    0.0285   -0.0242    0.0132   -0.0070    0.0035
    0.4400   -0.9273    0.0943   -0.0356    0.0307   -0.0110    0.0051   -0.0023    0.0011
    0.4400    1.0000   -0.1442    0.0306   -0.0452    0.0170   -0.0088    0.0040   -0.0021
    0.4400    0.2188   -0.0285    0.0020   -0.0035    0.0001    0.0001   -0.0001    0.0001]



#+ compare the performance of the MPC and NN policies
#* assign a penalty of 1000*slack for violation of constraints

function comp_bd_penal(x)
	return 1000*(sum(abs, (max.(0.,x_lbd .- x))) + sum(abs, (max.(0., x .- x_upd))))
end

function comp_disc_system!(xnp1,xn,policy,t)
	xview = @view xn[1:nx]
	u = policy(xview)

	xnp1[1:nx] = Ad * xview + Bd * u

	xnp1[obj_indx] = xn[obj_indx] + xview'*Q*xview + u'*R*u
	xnp1[con_indx] = xn[con_indx] + comp_bd_penal(xview)
	nothing
end

include(scriptsdir("6_LTI","jump_mpc.jl"))
_nn_policy = x -> nn(x,NN_params)

x0 = [ones(nx)*0.44; 0.e0; 0.e0]

x0 = [rand(nx)*0.44; 0.e0; 0.e0]
x0 = [0.4153165326230863, 0.18183460352788416, 0.32320559879596295, 0.44, 0.0933256890359769, -0.44, 0.0, 0.0]
x0 = [-0.44, 0.22041611218023704, 0.32562131430838825, 0.4287156014212751, 0.44, 0.33540356354489553, 0.0, 0.0]

x0[6]=-0.44
policy_prob = ODEProblem((dx,x,p,t)-> comp_disc_system!(dx,x,_nn_policy,t), x0, tspan)
mpc_prob = ODEProblem((dx,x,p,t)-> comp_disc_system!(dx,x,solve_mpc,t), x0, tspan)

println("Checking setup")

ode_alg = FunctionMap()
@time sol_nn = solve(policy_prob, ode_alg)
@time sol_mpc = solve(mpc_prob, ode_alg)
sol_nn(8)[obj_indx]
sol_mpc(8)[obj_indx]


mpc_pl =	plot(t_point,Array(sol_mpc)[1:6,:]' ; line_style... )
savefig(mpc_pl, plotsdir("LTI_6/lti_6_mpc_traj.svg"))
policy_pl = plot(t_point,Array(sol_nn)[1:6,:]' ; line_style...)
savefig(policy_pl, plotsdir("LTI_6/lti_6_nn_traj.svg"))


function obs_comp(sol)
	x = @view sol[1:nx,end]
	return sol[obj_indx,end]+sol[con_indx,end]+ comp_bd_penal(x)
end
function E_comp(prob, tol=1e-2)
	e = expectation(obs_comp, prob, [x0_dist; 0.e0; 0.e0], 0., Koopman(), ode_alg;
	quadalg = quad_alg, iabstol=tol,ireltol=tol)[1]
	return e
end
quad_alg = IntegralsCuba.CubaSUAVE(); 
E_comp(policy_prob, 1e-3)
E_comp(mpc_prob, 1e-3)
