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
xs = [0.4;0.6]

#+ Define controller to use
# hardtanh relu sigmoid 
Width = 12
act = sigmoid
clmp = sigmoid
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

const u_norm = 0.3 

function nn(x,θ)
	u = u_norm.* (clmp.(re(x,θ)[1])) #
	return u
end
n_weights=length(p_nn)
println("Number of weights: $n_weights")

function con_violate(x)
	return max(-x[2],0.) + max(x[2]-1.,0) +  max(x[1]-1.,0) +max(-x[1],0.)
end
Δt = 0.5
function system!(dx,x,_p,t)
	ψc = 0.4
	B = _p[2]
	#W = _p[3]
	θ = @view _p[3:end]
	xview = @view x[1:2]
	u = nn(xview,θ)

	#B=1
	H=0.18
	# ψc = 0.5
	
	W=0.25
	_γ=_p[1]

	ψe = ψc + H*(1+1.5*(x[1]/W -1) - 0.5*(x[1]/W -1)^3)
	dx[1] = B*(ψe-x[2]-u)
	dx[2] = (1/B)*(x[1]-_γ*sqrt(abs(x[2]))*sign(x[2]))
	dx[3] = (100*sum(abs2,xview .-xs) + 8*u*u + 800*max(-x[2]+0.4,0.)^2)
	dx[4] = con_violate(xview)
	nothing
end
const p_dis = [truncated(Normal(0.5, 0.017),0.45,0.55),truncated(Normal(.85, 0.1),0.7,1.)]
#+ Problem set up
const x0_dist =  [Uniform(0.2, 1.e0),Uniform(0., 0.7)] 
x0 = Float64[0.2;0.7; 0.e0; 0.e0]
const x_dim =2
#+ timepoints

const t0 = 0.0e0
const tf = 6.e0 
const tspan = (t0, tf)

ppp = rand.(p_dis)
#+ check set-up
prob = ODEProblem(system!, x0, tspan, [ppp;p_nn],
#adaptive=false,dt=Δt
);
# save only at final time point
prob_exp = ODEProblem(system!, x0, tspan, [ppp;p_nn],  save_everystep=false
#,adaptive=false,dt=Δt
) ;
ode_alg = AutoTsit5(Rosenbrock23()); #Rodas4P()#

println("Checking ODE setup");

@time sol = solve(prob_exp, ode_alg,save_everystep=false,)

const x_lb = [x0_dist[i].a for i in 1:x_dim]
const x_ub = [x0_dist[i].b for i in 1:x_dim]
const p_lb = [p_dis[i].lower for i in 1:length(p_dis)]
const p_ub = [p_dis[i].upper for i in 1:length(p_dis)]
lb=[x_lb; p_lb]
ub=[x_ub; p_ub]
sobol_samples = SobolSeq(lb, ub)
n_s = 100
skip(sobol_samples, n_s) # some authors suggest better uniformity by first skipping the initial portion of the LDS
xp_samples = reduce(hcat, next!(sobol_samples) for i = 1:n_s)'
x0_samples = xp_samples[:,1:x_dim]
p_samples = xp_samples[:,x_dim+1:end]


function obs_loss_x0(x0, p_nn, ρ, λ)
	sol = solve(remake(prob_exp, u0=[x0;0.e0; 0.e0],p=p_nn), ode_alg,save_everystep=false)
	h = sol[4,end] + con_violate(sol[1:2,end])
	return sol[3,end] + 0.5*ρ*h .^2 - λ*h
end

function cb(p_nn, l)
	println("Current loss is: $l")
	false
end
function rand_cb(p_nn, l)
	println("Current loss is: $(printEV_loss(p_nn))")
	false
end
function cb_iter(p_nn, l, k)
	println("Iteration $k, loss: $l")
	false
end

function obs_loss(sol, ρ, λ)
	h = sol[4,end] + con_violate(sol[1:2,end])
	return [sol[3,end] + 0.5*ρ*h .^2 - λ*h ;(sol[3,end] + 0.5*ρ*h .^2 - λ*h)^2]
end
function obs_con(sol)
		h = sol[4,end] + con_violate(sol[1:2,end])
		return h
end
function E_con(p_nn,prob)
	e = expectation(obs_con, prob, [x0_dist; 0.e0; 0.e0], [p_dis; p_nn], Koopman(), ode_alg;
	quadalg =IntegralsCubature.HCubatureJL(), iabstol=1e-4,ireltol=1e-4)[1]
	return e
end
const tol = 1e-2
# integral options
#IntegralsCuba.CubaSUAVE() # IntegralsCubature.HCubatureJL() # IntegralsCuba.CubaCuhre() IntegralsCubature.CubatureJLp
function EV_loss(p_nn, param, obs_loss;ρ_EV=.15)
	e = expectation(obs_loss, param[1], [x0_dist; 0.e0; 0.e0], [p_dis; p_nn], Koopman(), ode_alg;
	quadalg =IntegralsCubature.HCubatureJL(), iabstol=param[2],ireltol=param[2], seed = rand(Int)).u
	EV = e[1] + ρ_EV*(e[2]-e[1]^2)
	return EV
end

function output_func(sol,i, ρ, λ) 
	h = sol[4,end] + con_violate(sol[1:2,end])
	if sol.t[end] != tf
		return (Inf,false)
	else

	return (sol[3,end] + 0.5*ρ*h .^2 - λ*h, false)
	end
end

function rand_loss(p_nn, _output_func)
	 # x0_samples[rand(1:n_s),:] next!(s)
	function prob_func(prob,i,repeat) 
		indx = rand(1:n_s)
		return remake(prob_exp, p=[p_samples[indx,:]; p_nn], u0 = [x0_samples[indx,:]; 0.e0; 0.e0],tspan=_tspan) 
	end

	ensemble_prob = EnsembleProblem(prob_exp, prob_func=prob_func, 
	output_func=_output_func,)
	sim = solve(ensemble_prob, ode_alg, EnsembleThreads(),trajectories=1)
	return mean(sim)

end

function run_opt(solver, opt_f, max_iters, ini_guess, prob_exp)
	opt_prob = OptimizationProblem(opt_f, ini_guess, prob_exp)
	opt_sol = solve(opt_prob, solver, maxiters = max_iters, progress = true, callback = cb,
	#abstol=tol,reltol=tol
	)
	return opt_sol
end
const _tspan=(t0,tf)


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
		if k_max>1
			h = E_con(opt_sol.u, prob_exp)
		end
		if h[1] < con_tol
			println("Constraint satisfied.")
			return opt_sol,h,ρ,λ
		else
			println("Constraint violation: $(h[1]) with ρ: $ρ and λ: $λ")
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
const _prob_exp = ODEProblem(system!, x0, _tspan, p_nn, save_everystep=false) 



printing_loss(sol) = obs_loss(sol, 0., 0.)
printEV_loss(x) = EV_loss(x,  (prob_exp, 1e-2),printing_loss)

println("Loss before optimization: $(printEV_loss(p_nn))")
throttle_cb = Flux.throttle(cb,2)
throttle_rand = Flux.throttle(rand_cb,90)


println("Starting optimization with AMSGrad")
opt_sol,hh,ρ,λ  = aug_Lag_random(rand_loss, prob_exp, p_nn, Optimisers.AMSGrad(); k_max= 10, inner_max_iter=2000,con_tol=0.01)


println("Loss after AMSGrad: $(printEV_loss(opt_sol.u))")
println("Plotting policy")

x1s = range(lb[1], ub[1], length=100)
x2s = range(lb[2], ub[2], length=100)
k1(x1,x2) =nn([x1,x2],opt_sol.u)[1]
cont_plot = contourf(x1s, x2s, k1, color=:viridis)
savefig(cont_plot, plotsdir("EV_compressor.png"))

NN_params = opt_sol.u
save_params = @strdict NN_params controller u_norm clmp

println("Saving NN parameters")

file_name = datadir("compressor/nn__EV__intermed")*".jld2"
save(file_name, save_params) 

#test EV_loss(opt_sol.u, prob_exp)
# EV_loss(x) = EV_loss(x, prob_exp)
# ForwardDiff.gradient(EV_loss, opt_sol.u)

# using Zygote
# Zygote.gradient(EV_loss, opt_sol.u)
# options: AutoReverseDiff(compile=false), AutoForwardDiff(), 

function aug_Lag_E(EV_loss, obs_loss, ini_guess, solver, solver_param,prob_exp; k_max::Int= 3,	λ::Real = 0., ρ::Real=0.5, γ::Real=2, con_tol::Real=1e-4, inner_max_iter::Int=100)
	p0 = copy(ini_guess)
	_obs(sol) = obs_loss(sol, ρ, λ)
	_EV_loss(x,p) = EV_loss(x,p, _obs)

	opt_f = OptimizationFunction(_EV_loss, Optimization.AutoForwardDiff())
	opt_prob = OptimizationProblem(opt_f, p0, solver_param)

	h=-1.

	for k in 1 : k_max
		println("Iteration: $k")
		_obs(sol) = obs_loss(sol, ρ, λ)
		_EV_loss(x,p) = EV_loss(x,p, _obs)
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

const _param = (prob_exp, 1e-1);

param_vec = opt_sol.u
opt_sol,hh,_ρ,_λ  = aug_Lag_E(EV_loss, obs_loss, copy(param_vec), Optimisers.AMSGrad(), _param,prob_exp; λ=λ, ρ=ρ, inner_max_iter=200,con_tol=0.01)

println("Loss after AMSGrad: $(printEV_loss(opt_sol.u))")
ρ,λ =_ρ,_λ

# finish with LD_LBFGS
const _param = (prob_exp, 1e-1);
param_vec = opt_sol.u;
opt_sol,hh,_ρ,_λ  = aug_Lag_E(EV_loss, obs_loss, copy(param_vec), NLopt.LD_LBFGS(), _param,prob_exp; λ=λ, ρ=ρ, inner_max_iter=20, k_max=10)


const _param = (prob_exp, 1e-1);

param_vec = opt_sol.u
opt_sol,hh,_ρ,_λ  = aug_Lag_E(EV_loss, obs_loss, copy(param_vec), Optimisers.AMSGrad(), _param,prob_exp; λ=λ, ρ=ρ, inner_max_iter=200,con_tol=0.01)

println("Loss after AMSGrad: $(printEV_loss(opt_sol.u))")
ρ,λ =_ρ,_λ

# finish with LD_LBFGS
const _param = (prob_exp, 1e-2);
param_vec = opt_sol.u;
opt_sol,hh,_ρ,_λ  = aug_Lag_E(EV_loss, obs_loss, param_vec, NLopt.LD_LBFGS(), _param,prob_exp; λ=λ, ρ=ρ, inner_max_iter=20)
# NLopt.LD_MMA NLopt.LD_LBFGS NLopt.LD_SLSQP Optimisers.AMSGrad()
println("Loss after LBFGS: $(printEV_loss(opt_sol.u))")
obj  = printEV_loss(opt_sol.u)
NN_params = opt_sol.u
# create a Dict with the relevant simulation data
save_params = @strdict NN_params controller u_norm clmp ρ λ 

println("Saving NN parameters")

file_name = datadir("compressor/nn__EV_015")*".jld2"
save(file_name, save_params) 

#=
const param = (prob_exp, 1e-4)
opt_sol = run_opt(NLopt.LD_LBFGS, opt_f, 20, opt_sol.u, param;) 
=#
# NLopt.LD_MMA NLopt.LD_LBFGS NLopt.LD_SLSQP Optimisers.AMSGrad()
#@time ode_sol = solve(prob, ode_alg, p=NN_params, u0 = Float64[rand.(x0_dist); 0.], tspan=(0,tf))

#plot(ode_sol)
#=
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
ensemblesol = solve(ensemble_prob,ode_alg,EnsembleThreads(),trajectories=250)

summ = EnsembleSummary(ensemblesol;quantiles = [0.0, 1.])
ens_plot = plot(summ,labels=["u1 95%" "u2 95%"],legend=:outertopright,idxs=1:2)
=#
println("Plotting policy")

k1(x1,x2) =  nn([x1,x2],opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis)

# save figures
savefig(cont_plot, plotsdir("cstr_nn_policy.png"))