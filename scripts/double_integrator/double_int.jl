using DrWatson 
@quickactivate "nn_controller"
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqUncertainty 
using Optimization, OptimizationOptimisers, OptimizationNLopt, OptimizationOptimJL
using Flux
using Distributions
using Plots
using ForwardDiff, NLopt
using LinearAlgebra
using IntegralsCuba, IntegralsCubature
using Sobol
gr()

# plotlyjs(legend=:outertopright,guidefont = 14, legendfontsize = 14,tickfont = 10,fg_legend = :transparent)

#+ Define controller to use
# hardtanh relu sigmoid 
Width = 10
act = relu
clmp = hardtanh
controller = f64(Flux.Chain(
Flux.Dense(2, Width, act),
Flux.Dense(Width, Width, act),
#Flux.Dense(Width, Width, act),
#Flux.Dense(Width, Width, act),
#Flux.Dense(Width, Width, act),
#Flux.Dense(Width, Width, act),
Flux.Dense(Width, 1 ))) # Note had to make it to accept float 64 iwth f64
p_nn,re = Flux.destructure(controller) # destructure  into a vector of parameters
n_weights = length(p_nn)

function nn(x,θ)
	x_scaling = [18; 6]
	u = u_norm * (clmp(re((x./x_scaling .+0.5),θ)[1]-re(([0;0]./x_scaling .+0.5),θ)[1])) #
	return u
end
println("Number of weights: $n_weights")


const u_norm = 3.e0
const P = [2.177565296080891 1.2571126613074837; 1.2571126613074837 1.480332243219584]


const Ad = Float64[1. 1.; 0. 1.]
const Bd = Float64[0.; 1.]
function disc_system!(xnp1,xn,θ,t)
	xview = @view xn[1:2]
	u = nn(xview,θ)

	xnp1[1:2] = Ad*xview + Bd*u
	xnp1[3] = xn[3] + (xn[1]*xn[1] + 0.05*xn[2]*xn[2]+ 0.1*u*u)
	xnp1[4] = xn[4] + max(0.,xn[2]-3.) + max(0.,-xn[2]-3.)
	nothing
end

system!(xnp1,xn,θ,t) = disc_system!(xnp1,xn,θ,t) 
#+ Problem set up
const x0_dist =  [Uniform(-9.e0, 9.e0),Uniform(-3.e0, 3.e0)] 


x0 = Float64[rand.(x0_dist); 0.e0; 0.e0]
const x_dim =2
#+ timepoints

const t0 = 0.0e0
const tf = 4.e0 
const tspan = (t0, tf)
# tsteps = t0:Δt:tf
# tlength=length(tsteps)

#+ check set-up
# prob = ODEProblem(system!, x0, tspan, p_nn, abstol=1e-5, reltol=1e-5)
prob = DiscreteProblem(disc_system!, x0, tspan, p_nn)				
# save only at final time point
# prob_exp = ODEProblem(system!, x0, tspan, p_nn,  abstol=1e-5, reltol=1e-5,save_everystep=false) 
prob_exp = DiscreteProblem(disc_system!, x0, tspan, p_nn,save_everystep=false)	 

println("Checking ODE setup")
#ode_alg = AutoTsit5(Rosenbrock23())
ode_alg = FunctionMap()
#@time sol = solve(prob, ode_alg,save_everystep=false)
@time sol = solve(prob,ode_alg,save_everystep=false)
# plot(sol[1:2,:],label=["x1" "x2"])

# CubaCuhre HCubatureJL
#using IntegralsCuba, IntegralsCubature.HCubatureJL, IntegralsCubature.CubatureJLp IntegralsCuba.Suave()
#  CubatureJLp IntegralsCuba.CubaCuhre() IntegralsCubature.HCubatureJL


# create sobol samples from x0_dist
const lb = [x0_dist[i].a for i in 1:x_dim]
const ub = [x0_dist[i].b for i in 1:x_dim]
sobol_samples = SobolSeq(lb, ub)

n_s = 100
skip(sobol_samples, n_s) # some authors suggest better uniformity by first skipping the initial portion of the LDS
x0_samples = reduce(hcat, next!(sobol_samples) for i = 1:n_s)'


function obs_loss_x0(x0, p_nn, ρ, λ)
	sol = solve(remake(_prob_exp, u0=[x0;0.e0; 0.e0],p=p_nn), ode_alg,save_everystep=false)
	h = sol[4,end] + max(0.,sol[2,end]-3.) + max(0.,-sol[2,end]-3.)
	return sol[3,end] + sol[1:2,end]'*P*sol[1:2,end] + 0.5*ρ*h .^2 - λ*h
end

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
	h = sol[4,end] + max(0.,sol[2,end]-3.) + max(0.,-sol[2,end]-3.)
	return sol[3,end] + sol[1:2,end]'*P*sol[1:2,end] + 0.5*ρ*h .^2 - λ*h
end
function obs_con(sol)
	return sol[4,end]+ max(0.,sol[2,end]-3.) + max(0.,-sol[2,end]-3.)
end
function E_con(p_nn,prob)
	e = expectation(obs_con, prob, [x0_dist; 0.e0; 0.e0], p_nn, Koopman(), ode_alg;
	quadalg =IntegralsCubature.HCubatureJL(), iabstol=1e-3,ireltol=1e-3)[1]
	return e
end
const tol = 1e-2
#IntegralsCuba.CubaSUAVE() # IntegralsCubature.HCubatureJL() # IntegralsCuba.CubaCuhre() IntegralsCubature.CubatureJLp
function E_loss(p_nn, param, obs_loss)
	e = expectation(obs_loss, param[1], [x0_dist; 0.e0; 0.e0], p_nn, Koopman(), ode_alg;
	quadalg =IntegralsCubature.HCubatureJL(), iabstol=param[2],ireltol=param[2], seed = rand(Int))[1]
	return e  + 10*abs(nn([0;0],p_nn))
end

output_func(sol,i, ρ, λ) = (sol[3,end]+ sol[1:2,end]'*P*sol[1:2,end] + 0.5*ρ*sol[4,end] .^2 - λ*sol[4,end], false)

function rand_loss(p_nn, _output_func)
	 # x0_samples[rand(1:n_s),:] next!(s)
	prob_func(prob,i,repeat) = remake(_prob_exp, p=p_nn, u0 = [x0_samples[rand(1:n_s),:]; 0.e0; 0.e0],tspan=_tspan) 
	ensemble_prob = EnsembleProblem(prob_exp, prob_func=prob_func, 
	output_func=_output_func,)
	sim = solve(ensemble_prob, ode_alg, EnsembleThreads(),trajectories=1)
	return mean(sim) + 10*abs(nn([0;0],p_nn))
end

function run_opt(solver, opt_f, max_iters, ini_guess, _prob_exp)
	opt_prob = OptimizationProblem(opt_f, ini_guess, _prob_exp)
	opt_sol = solve(opt_prob, solver, maxiters = max_iters, progress = true, callback = cb,
	)
	return opt_sol
end
const _tspan=(t0,tf)
const _prob_exp = ODEProblem(system!, x0, _tspan, p_nn,  abstol=1e-6, reltol=1e-6,save_everystep=false) 

# options: AutoReverseDiff(compile=false), AutoForwardDiff(), 
function aug_Lag_random(loss_func, prob_exp, ini_guess, solver; k_max::Int= 3,	λ::Real = 0., ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
	_output_func(sol,i) = output_func(sol,i, ρ, λ)
	opt_rand = OptimizationFunction(loss_func, Optimization. AutoForwardDiff())
	opt_prob = OptimizationProblem(opt_rand, p_nn, _output_func)
	h=-1.

	for k in 1 : k_max
		println("Iteration: $k")
		_output_func(sol,i) = output_func(sol,i, ρ, λ)
		_opt_problem = remake(opt_prob,u0=ini_guess, p =_output_func)
		opt_sol = solve(_opt_problem, solver, maxiters = inner_max_iter, progress = true, callback = throttle_rand,save_best=false)
		h = E_con(opt_sol.u, prob_exp)
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

printing_loss(sol) = obs_loss(sol, 0., 0.)
printE_loss(x) = E_loss(x,  (_prob_exp, 1e-4),printing_loss)

println("Loss before optimization: $(printE_loss(p_nn))")
throttle_cb = Flux.throttle(cb,2)
throttle_rand = Flux.throttle(rand_cb,5)

#+ should be ~82, after optimization

println("Starting optimization with AMSGrad")
opt_sol,h,ρ,λ  = aug_Lag_random(rand_loss, _prob_exp, p_nn, Optimisers.AMSGrad(); k_max= 50,	λ=λ, ρ=ρ,inner_max_iter=6000)
#opt_sol = solve(opt_prob, Optimisers.AMSGrad(), maxiters = 3000, progress = true, callback = throttle_rand ,save_best=false)

println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")
println("Plotting policy")

levels = range(-u_norm,u_norm,step=0.1)
x1s = range(lb[1], ub[1], length=200)
x2s = range(lb[2], ub[2], length=200)

k1(x1,x2) = nn([x1,x2],opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis,levels=levels)
savefig(cont_plot, plotsdir("disc_double_int_nn_policy_AMSgrad.png"))

NN_params = opt_sol.u
save_params = @strdict NN_params controller u_norm clmp

println("Saving NN parameters")

file_name = datadir("double_int/disc_nn__intermed")*".jld2"
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
		h = E_con(opt_sol.u, prob_exp)
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

const _param = (_prob_exp, 1e-3);

param_vec = opt_sol.u
opt_sol,h,_ρ,_λ  = aug_Lag_E(E_loss, obs_loss, param_vec, Optimisers.AMSGrad(1f-4), _param,_prob_exp; λ=λ, ρ=ρ, inner_max_iter=400)

println("Loss after AMSGrad: $(printE_loss(opt_sol.u))")
ρ,λ =_ρ,_λ

# finish with LD_LBFGS
const _param = (_prob_exp, 1e-4);
param_vec = opt_sol.u;
opt_sol,h,_ρ,_λ  = aug_Lag_E(E_loss, obs_loss, param_vec, NLopt.LD_LBFGS(), _param,_prob_exp; λ=λ, ρ=ρ, inner_max_iter=20)

# NLopt.LD_MMA NLopt.LD_LBFGS NLopt.LD_SLSQP Optimisers.AMSGrad()
println("Loss after LBFGS: $(printE_loss(opt_sol.u))")
obj  = printE_loss(opt_sol.u)
NN_params = opt_sol.u
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller u_norm clmp ρ λ 

println("Saving NN parameters")

file_name = datadir("double_int/t_disc_nn")*".jld2"
save(file_name, save_params) 

println("Plotting policy")

k1(x1,x2) =  nn([x1,x2],opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis,levels=levels)

# save figures
savefig(cont_plot, plotsdir("disc_double_int_nn_policy.png"))
