using DrWatson 
@quickactivate "nn_controller"
using Flux, JuMP, Ipopt, MLUtils
using Plots
using OrdinaryDiffEq, Distributions, Cubature
using Sobol
using Optimization, OptimizationNLopt, OptimizationOptimisers
plotlyjs()

# using the same architecture and sample points as in double-int.jl 
# fit a NN to the MPC policy

#+ Define controller to use
# hardtanh relu sigmoid 
Width = 12
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
	u = u_norm * (clmp(re((x./x_scaling .+0.5),θ)[1] 
	-re(([0;0]./x_scaling .+0.5),θ)[1])) #
	return u
end
println("Number of weights: $n_weights")

const u_norm = 3.e0

const x0_dist =  [Uniform(-9.e0, 9.e0),Uniform(-3.e0, 3.e0)] #[Uniform(-1., 1.);Uniform(-1., 1.)]
const lb = [x0_dist[i].a for i in 1:2]
const ub = [x0_dist[i].b for i in 1:2]
sobol_samples = SobolSeq(lb, ub)

n_s = 100
skip(sobol_samples, n_s) # some authors suggest better uniformity by first skipping the initial portion of the LDS
x0_samples = reduce(hcat, next!(sobol_samples) for i = 1:n_s)'
# scatter(x0_samples[:,1], x0_samples[:,2])

include("scripts/double_integrator/jump_mpc.jl")
u_mpc = zeros(n_s)
for i in 1:n_s
	u_mpc[i] = solve_mpc(x0_samples[i,:])
end

# define loss function

# only written for batch size of 1
function flux_loss(controller,x,u)
	x_scaling = [18; 6]
	hat_u =  u_norm*controller((x./x_scaling .+0.5))
	loss = sum(abs2,u- hat_u)
	return loss
end

data_loader = DataLoader((x0_samples', u_mpc), batchsize=1, shuffle=true);


opt_state = Flux.setup(Flux.AMSGrad(), controller) 

num_epochs = 200

for k=1:num_epochs
	Flux.Optimise.train!(flux_loss, controller, data_loader, opt_state)
end


p_nn,re = Flux.destructure(controller) # destructure  into a vector of parameters

NN_params = p_nn
# now check the loss for all the data 

function loss(p_nn,a=nothing)
	x_scaling = [18; 6]
	loss = 0
	wrap(x) =  u_norm * (clmp(re((x./x_scaling .+0.5),p_nn)[1])) #
	for i=1:n_s
		hat_u = wrap(x0_samples[i,:])[1]
		loss += (u_mpc[i] - hat_u)^2
	end
	return loss
end
loss(p_nn)

opt_f = OptimizationFunction(loss, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_f, p_nn)

loss(p_nn)
opt_sol = solve(opt_prob, Optimisers.AMSGrad(), maxiters = 5000, progress = true)
loss(opt_sol.u)
opt_prob = OptimizationProblem(opt_f, opt_sol.u)
opt_sol = solve(opt_prob, NLopt.LD_LBFGS(), maxiters = 20, progress = true)
loss(opt_sol.u)


levels = range(-u_norm,u_norm,step=0.1)
x1s = range(lb[1], ub[1], length=200)
x2s = range(lb[2], ub[2], length=200)
#k1(x1,x2) = u_norm*clmp.(re([x1,x2],opt_sol.u) -re(Float64[0.0, 0.0], opt_sol.u))[1]
k1(x1,x2) = nn([x1,x2],opt_sol.u)
cont_plot = contourf(x1s, x2s, k1, color=:viridis,levels=levels)
savefig(cont_plot, plotsdir("disc_double_int_imitation.png"))

NN_params = opt_sol.u


save_params = @strdict NN_params controller u_norm clmp


println("Saving NN parameters")
file_name = datadir("double_int/imitation_lbfgs")*".jld2"
save(file_name, save_params) 
