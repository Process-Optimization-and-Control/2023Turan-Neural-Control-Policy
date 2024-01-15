using DrWatson
@quickactivate "nn_controller"
using Flux, JuMP, Ipopt
using Plots, Plots.Measures
using OrdinaryDiffEq, Distributions, Cubature
using LaTeXStrings

gr()
Plots.reset_defaults()  

default(xlabel=L"x_1", ylabel=L"x_2",colorbar_titlefontsize=20,labelfontsize=20,tickfontsize=10, legendfontsize=14)
cbar_options = (colorbar_titlefontsize = 13,rightmargin=5mm)

# definitions from double_int.jl

nn_file = datadir("par_double_int/disc_nn_hrdtanh")*".jld2" # embedded optimization

imi_nn_file = datadir("double_int/imitation_lbfgs")*".jld2"
imi_strut  = load(imi_nn_file)
imi_params = imi_strut["NN_params"]
imi_controller = imi_strut["controller"]

nn_strut = load(nn_file)
NN_params = nn_strut["NN_params"]
controller = nn_strut["controller"]
clmp = nn_strut["clmp"]
u_norm = nn_strut["u_norm"]


_,re = Flux.destructure(controller) # destructure  into a vector of parameters
_, imi_re = Flux.destructure(imi_controller)

len = 100 # reasonably small for the MPC solves
x1s = collect(range(-9, 9., length=len) )
x2s = collect(range(-3, 3., length=len))

points = [[x1,x2] for x1 in x1s, x2 in x2s]

#nn_policy(x1,x2) = u_norm*clmp.(re([x1,x2],NN_params) -re(Float64[0.0, 0.0], NN_params))[1]
const x_scaling = [18; 8]
const origin = ([0; 0] ./ x_scaling .+ 0.5)

function par_nn(x,θ;pcon=3)
	xx = (x[1:2] ./ x_scaling .+ 0.5)
    _p = pcon - 3.0
    u = u_norm * (clmp(re([xx; _p], θ)[1] - re([origin; _p], θ)[1])) # clmp = hardtanh = mid
    return u
end

function nn(x,θ)
	x_scaling = [18; 6]
	u = u_norm * (clmp(imi_re((x./x_scaling .+0.5),θ)[1]-imi_re(([0;0]./x_scaling .+0.5),θ)[1])) #
	return u
end
nn_policy_4(x1,x2) = par_nn([x1;x2],NN_params, pcon=4.)

nn_policy(x1,x2) = par_nn([x1;x2],NN_params, pcon=3.)
imi_policy(x1,x2) = nn([x1;x2],imi_params)

u_nn = @. nn_policy(x1s',x2s)
u_nn4 = @. nn_policy_4(x1s',x2s)
u_imi = @. imi_policy(x1s',x2s)

isapprox(nn_policy(0.,0.) , 0, atol=1e-6)

include("/home/evren/Documents/2023/NN controller/scripts/double_integrator/jump_mpc_4.jl")
mpc_policy(x1,x2) = solve_mpc([x1,x2])
open_policy(x1,x2) = solve_open_loop([x1,x2])
u_mpc = @.  mpc_policy(x1s',x2s)

u_diff = (u_nn-u_mpc)
abs_diff = abs.(u_diff)

imi_diff = (u_imi-u_mpc)
abs_imi_diff = abs.(imi_diff)

# plot the difference

diff_plot = contourf(x1s, x2s, u_diff, color=:viridis,colorbartitle=L"u_{nn}-u_{mpc}";cbar_options...)
savefig(diff_plot, plotsdir("abs_diff_policy_double_int.svg"))

levels = range(-u_norm,u_norm,step=0.1)

# plot the NN policy
nn_contf = contourf(x1s, x2s, nn_policy, color=:viridis,levels=levels,colorbartitle=L"u")
savefig(nn_contf, plotsdir("par_double_int","nn_policy_double_int.svg"))
nn_contf4 = contourf(x1s, x2s, nn_policy_4, color=:viridis,levels=levels,colorbartitle=L"u")
savefig(nn_contf4, plotsdir("par_double_int","nn_policy_double_int_4.svg"))

u_mpc[u_mpc .>= 3.0] .= 3.0
u_mpc[u_mpc .<= -3.0] .= -3.0

mpc_contf = contourf(x1s, x2s, u_mpc, color=:viridis,levels=levels,colorbartitle=L"u")
savefig(mpc_contf, plotsdir("par_double_int","mpc_double_int_4.svg"))

# and the system

const Ad = Float64[1. 1.; 0. 1.]
const Bd = Float64[0.; 1.]
const P = [2.177565296080891 1.2571126613074837; 1.2571126613074837 1.480332243219584]
# const weights = [1. 0; 0 1]

function __nn_system!(xnp1,xn,θ,t,policy)
	u = policy(xn[1],xn[2])
	xview = @view xn[1:2]
	xnp1[1:2] = Ad*xview + Bd*u

	xnp1[3] = xn[3] + (xn[1]*xn[1] + 0.05*xn[2]*xn[2]+ 0.1*u*u)
	nothing
end
nn_system!(xnp1,xn,θ,t) = __nn_system!(xnp1,xn,θ,t,nn_policy)
imi_system!(xnp1,xn,θ,t) = __nn_system!(xnp1,xn,θ,t,imi_policy)

function mpc_system!(xnp1,xn,θ,t)
	xview = @view xn[1:2]
	u = mpc_policy(xn[1],xn[2])
	xnp1[1:2] = Ad*xview + Bd*u

	xnp1[3] = xn[3] + (xn[1]*xn[1] + 0.05*xn[2]*xn[2]+ 0.1*u*u)
	nothing
end

#+ Problem set up
const __x0_dist =  [Uniform(-9.e0, 9.e0),Uniform(-3.e0, 3.e0)] 

__x0 = Float64[9.;2.9; 0.e0]
#+ timepoints

const _t0 = 0.0e0
const _tf = 4.e0 
const __tspan = (_t0, _tf)
ode_alg = FunctionMap()
# tsteps = t0:Δt:tf
# tlength=length(tsteps)

#+ check set-up
# prob = ODEProblem(system!, x0, tspan, p_nn, abstol=1e-5, reltol=1e-5)
mpc_prob = DiscreteProblem(mpc_system!, __x0, __tspan)				
nn_prob = DiscreteProblem(nn_system!, __x0, __tspan)
# save only at final time point
@time _sol = solve(mpc_prob, ode_alg)
@time _sol = solve(nn_prob, ode_alg)

function open_loop_mpc(x0)
	x = zeros(2,4)
	x[:,1] = x0
	u = open_policy(x0[1],x0[2])
	loss = 0.
	for i in 1:3
		x[:,i+1] = Ad*x[:,i] + Bd*u[i]
		loss += (x[1,i]*x[1,i] + 0.05*x[2,i]*x[2,i]+ 0.1*u[i]*u[i])
	end
	if any(x[2,:] .> 3 + 1e-6) || any(x[2,:] .< -3 - 1e-6)
		return Inf
	end
	return loss + x[1:2,end]'*P*x[1:2,end]
end
function obs_loss_mpc(x0)
	mpc_prob = DiscreteProblem(mpc_system!, [x0;0.], __tspan)
	sol = solve(mpc_prob, ode_alg)
	if any(sol[2,:] .> 3 + 1e-2) || any(sol[2,:] .< -3 - 1e-2)
		return Inf
	end
	return sol[3,end] + sol[1:2,end]'*P*sol[1:2,end]
end

function obs_loss_nn(x0)
	nn_prob = DiscreteProblem(nn_system!, [x0;0.], __tspan)
	sol = solve(nn_prob, ode_alg)
	
	if any(sol[2,:] .> 3 + .1) || any(sol[2,:] .< -3 - .1)
		return Inf
	end
	
	return sol[3,end] + sol[1:2,end]'*P*sol[1:2,end]
end
function obs_loss_imi(x0)
	imi_prob = DiscreteProblem(imi_system!, [x0;0.], __tspan)
	sol = solve(imi_prob, ode_alg)
	
	
	if any(sol[2,:] .> 3 + .1) || any(sol[2,:] .< -3 - .1)
		return Inf
	end
	
	return sol[3,end] + sol[1:2,end]'*P*sol[1:2,end]
end
# mpc_ol_perf(x1,x2) = open_loop_mpc([x1;x2])
mpc_cl_perf(x1, x2) = obs_loss_mpc([x1;x2])
nn_cl_perf(x1, x2) = obs_loss_nn([x1;x2])
imi_cl_perf(x1, x2) = obs_loss_imi([x1;x2])


#v_mpc_ol = @.  mpc_ol_perf(x1s',x2s)
v_mpc = @.  mpc_cl_perf(x1s',x2s)


plot_v_mpc = v_mpc
plot_v_mpc[plot_v_mpc .< 1e-6] .= 1e-6
mpc_v_contf = contourf(x1s, x2s, log10.(plot_v_mpc), color=:viridis,colorbartitle=L"\log_{10}V_{cl}",colorbar_titlefontsize = 14,rightmargin=10mm)
savefig(mpc_v_contf, plotsdir("v_mpc_policy_double_int.svg"))

v_nn = @.  nn_cl_perf(x1s',x2s) 
v_imi = @.  imi_cl_perf(x1s',x2s)

plot_v_nn = copy(v_nn)
plot_v_nn[plot_v_nn .< 1e-4] .= 1e-4

plot_v_nn2 = plot_v_nn
nn_v_contf = contourf(x1s, x2s, log10.(plot_v_nn), color=:viridis,colorbartitle=L"\log_{10}V_{NN}";size=(800,600),cbar_options...)
savefig(nn_v_contf, plotsdir("par_double_int","v_nn_policy_double_int.svg"))

v_diff = abs.(plot_v_nn-plot_v_mpc)
v_diff[(v_diff) .<=1e-4] .=1e-4
minimum(v_diff[v_diff .!=0.]) == 1e-4

diff_v_contf = contourf(x1s, x2s, log10.(v_diff),color=:viridis,
#levels=sort(collect(range(0.,-6,length=20)),rev=false),
levels=10, 
xlabel=L"x_1", ylabel=L"x_2",colorbartitle=L"\log_{10}\left(V_{NN}-V^{CL}_{mpc}\right)";size=(800,600),cbar_options...)
savefig(diff_v_contf, plotsdir("par_double_int","v_diff_nn_policy_double_int.svg"))

imi_diff = abs.(v_imi-plot_v_mpc)
imi_diff[imi_diff .< 1e-4] .= 1e-4

imi_diff_v_contf = contourf(x1s, x2s, log10.(imi_diff), color=:viridis, levels=10,
xlabel=L"x_1", ylabel=L"x_2",colorbartitle=L"\log_{10}\left(V_{NN,imi}-V^{CL}_{mpc}\right)";size=(800,600),cbar_options...)
savefig(imi_diff_v_contf, plotsdir("par_double_int","v_diff_imi_policy_double_int.svg"))

function integrand(x0)
	prod(pdf.(x0_dist, x0))*obs_loss_mpc(x0)
end
(val,err) = Cubature.hcubature(integrand, lb, ub, reltol=1e-3, abstol=1e-3, maxevals=0)
