using DrWatson
@quickactivate "nn_controller"
using Flux, JuMP, Ipopt
using Plots
using DifferentialEquations, Distributions, Cubature
using LaTeXStrings
#plotlyjs()

gr()
Plots.reset_defaults()

default(xlabel=L"x_1", ylabel=L"x_2", colorbar_titlefontsize=20, labelfontsize=20, tickfontsize=8, legendfontsize=14)

case_list = ["nn"; "nn__EV"; "nn__EV_001"; "nn__EV_005"; "nn__EV_015"; "nn__EV_1"]
case = case_list[5]
loaded = load(datadir("compressor/" * case) * ".jld2")

clmp = loaded["clmp"]
controller = loaded["controller"]
u_norm = loaded["u_norm"]
NN_params = loaded["NN_params"]
_, re = Flux.destructure(controller)
len = 50 # reasonably small for the MPC solves
x1s = collect(range(0.2, 1.0, length=len))
x2s = collect(range(0.0, 0.8, length=len))

function nn(x, θ)
    u = u_norm .* (clmp.(re(x, θ)[1])) #
    return u
end
nn_policy(x1, x2) = nn([x1, x2], NN_params)

u_level = 0:0.025:0.3

u_nn = @. nn_policy(x1s', x2s)
u_nn[u_nn.<1e-5] .= 0.0
u_nn[u_nn.>0.3-1e-5] .= 0.3 # to force the color bar
nn_contf = contourf(x1s, x2s, u_nn, color=:viridis, colorbar_title=L"u", levels=u_level)

savefig(nn_contf, plotsdir("nn_policy_comp" * case * ".svg"))


include(scriptsdir("compressor_surge/jump_mpc.jl"))
mpc_policy(x1, x2) = solve_mpc([x1, x2])[1]

do_mpc = false # avoid running the MPC parts, as they take time.

if do_mpc 

u_mpc = @. mpc_policy(x1s', x2s)
u_mpc[u_mpc.<1e-5] .= 0.0
u_mpc[u_mpc.>0.3-1e-5] .= 0.3 # to force the color bar
mpc_contf = contourf(x1s, x2s, u_mpc, color=:viridis, colorbar_title=L"u", levels=u_level)

savefig(mpc_contf, plotsdir("mpc_comp.svg"))
end

xs = [0.4; 0.6]

function con_violate(x)
    return max(-x[2], 0.0) + max(x[2] - 1.0, 0) + max(x[1] - 1.0, 0) + max(-x[1], 0.0)
end
function _system!(dx, x, _p, t, policy)
    ψc = 0.4
    B = _p[2]
    xview = @view x[1:2]
    if policy == nn_policy
        u = policy(xview...)
    else
        u = _p[3]
    end
    #B=1
    H = 0.18
    # ψc = 0.5
    W = 0.25
    _γ = _p[1]

    ψe = ψc + H * (1 + 1.5 * (x[1] / W - 1) - 0.5 * (x[1] / W - 1)^3)
    dx[1] = B * (ψe - x[2] - u)
    dx[2] = (1 / B) * (x[1] - _γ * sqrt(abs(x[2])) * sign(x[2]))
    dx[3] = (100 * sum(abs2, xview .- xs) + 8 * u * u + 800 * max(-x[2] + 0.4, 0.0)^2)
    dx[4] = con_violate(xview)

    nothing
end
const t0 = 0.0e0
const plot_tf = 30.e0
const plot_tspan = (t0, plot_tf)

dist_times = range(t0, plot_tf, step=3.0)
function dist_affect!(integrator)
    integrator.p[1:2] = rand.(p_dis)
end
times = range(t0, plot_tf, step=0.5)
function affect!(integrator)
    u = mpc_policy(integrator.u[1:2]...)
    if u != Inf
        integrator.p[3] = u
    else
        integrator.u[3] = 1e8
    end
end
cb_de = PresetTimeCallback(times, affect!)
cb_dist = PresetTimeCallback(dist_times, dist_affect!)

mpc_system!(dx, x, _p, t) = _system!(dx, x, _p, t, mpc_policy)
nn_system!(dx, x, _p, t) = _system!(dx, x, _p, t, nn_policy)

const p_dis = [truncated(Normal(0.5, 0.017), 0.45, 0.55), truncated(Normal(0.85, 0.1), 0.7, 1.0)]
#+ Problem set up
const x0_dist = [Uniform(0.2, 1.e0), Uniform(0.0, 0.7)] #[Uniform(-1., 1.);Uniform(-1., 1.)]
x0 = [0.8; 0.1; 0.0; 0]
ppp = [0.5; 0.85; 0.0]


pl_x0 = [[0.7; 0.1; 0.e0; 0.e0] [0.4; 0.4; 0.e0; 0.e0] [0.25; 0.1; 0.e0; 0.e0]]
pl_ppp = [[0.5; 0.85; 0.0] [0.45; 1.0; 0.0] [0.55; 0.7; 0.0]]#rand.(p_dis)

mpc_prob = ODEProblem(mpc_system!, x0, plot_tf, ppp, callback=cb_de);
nn_prob = ODEProblem(nn_system!, x0, plot_tf, ppp, saveat=0.1);
dist_prob = ODEProblem(nn_system!, [xs; 0; 0], plot_tf, ppp, callback=cb_dist);
ode_alg = AutoTsit5(Rosenbrock23()); #Rodas4P()#

alt_cont = deepcopy(nn_contf);


for c_x = 1:3
    for c_p = 1:2
        if c_p == 2
            sty = (color=:red, style=:dash,)
        elseif c_p == 3
            sty = (color=:orange, style=:dashdot,)
        else
            sty = (color=:pink,)
        end
        nn_prob = ODEProblem(nn_system!, pl_x0[:, c_x], plot_tf, pl_ppp[:, c_p])
        sol = solve(nn_prob, ode_alg)
        plot!(alt_cont, sol(0)[1, :], sol(0)[2, :], label="", linewidth=2, marker=:circle, markersize=5; sty...)
        plot!(alt_cont, sol(0:0.1:plot_tf)[1, :], sol(0:0.1:plot_tf)[2, :], label="", linewidth=2; sty...)
        if do_mpc
            mpc_prob = ODEProblem(mpc_system!, pl_x0[:, c_x], plot_tf, pl_ppp[:, c_p], callback=cb_de)
            sol = solve(mpc_prob, ode_alg)
            plot!(mpc_contf, sol(0)[1, :], sol(0)[2, :], label="", linewidth=2, marker=:circle, markersize=5; sty...)
            plot!(mpc_contf, sol(0:0.1:plot_tf)[1, :], sol(0:0.1:plot_tf)[2, :], label="", linewidth=2; sty...)
        end
    end
end
if do_mpc
    xlims!(mpc_contf, (0.2, 1.0))
    ylims!(mpc_contf, (0.0, 0.8))
end
xlims!(alt_cont, (0.2, 1.0))
ylims!(alt_cont, (0.0, 0.8))

savefig(alt_cont, plotsdir("nn_policy_traj" * case * ".svg"))
if do_mpc
    savefig(mpc_contf, plotsdir("mpc_policy_traj.svg"))
end

if do_mpc
	@time sol = solve(mpc_prob, ode_alg)
end
# @time sol = solve(dist_prob, ode_alg)

plot(0:0.1:plot_tf, sol(0:0.1:plot_tf)[1:2, :]', label="", color=[palette(:greens) palette(:reds)], width=2,)

sols_mpc = Vector{Any}(undef, 10);
sols_nn = Vector{Any}(undef, 10);

for i = 1:10
    println(i)
    pp = [rand.(p_dis); 0.0]
    if do_mpc
        mpc_prob = remake(mpc_prob, p=pp)
        sols_mpc[i] = solve(mpc_prob, ode_alg)
    end
    nn_prob = remake(nn_prob, p=pp)
    sols_nn[i] = solve(nn_prob, ode_alg)
end



cols = [cgrad(:greens, 10, categorical=true, scale=:log); cgrad(:reds, 10, categorical=true)]
style = (xlabel=L"t", ylabel=L"x",
    width=1.5, opacity=1.0);
pl_nn = plot(; style...);
pl_mpc = plot(; style...);

i = 1
plot!(pl_nn, 0.0:0.1:6.0, sols_nn[i](0.0:0.1:6.0)[1:2, :]', label=[L"x_1" L"x_2"], color=[cols[1][i] cols[2][i]], legend=:bottomright);
plot!(pl_mpc, 0.0:0.1:6.0, sols_mpc[i](0.0:0.1:6.0)[1:2, :]', label=[L"x_1" L"x_2"], color=[cols[1][i] cols[2][i]], legend=:bottomright);
for i = 2:10
    plot!(pl_nn, 0.0:0.1:6.0, sols_nn[i](0.0:0.1:6.0)[1:2, :]', label="", color=[cols[1][i] cols[2][i]])
    if do_mpc
        plot!(pl_mpc, 0.0:0.1:6.0, sols_mpc[i](0.0:0.1:6.0)[1:2, :]', label="", color=[cols[1][i] cols[2][i]])
    end
end
savefig(pl_nn, plotsdir("comp_traj_" * case * ".svg"))
if do_mpc
savefig(pl_mpc, plotsdir("comp_traj_mpc.svg"))
plot(pl_nn, pl_mpc, layout=(2, 1))
end


using DifferentialEquations.EnsembleAnalysis
nn_prob = ODEProblem(nn_system!, x0, plot_tf, ppp, saveat=0.1);
if do_mpc
	mpc_prob = ODEProblem(mpc_system!, x0, plot_tf, ppp, callback=cb_de, saveat=0.1);
end
# run ensemble of simulations

function prob_func(prob, i, repeat)
    @. prob.u0 = [rand.(x0_dist); 0; 0]
    prob.p[1:2] = rand.(p_dis)
    return prob
end
function ens_output_func(sol, i)
    return (sol[1:2, :], false)
end
num_traj = 3000

ens_prob = EnsembleProblem(nn_prob,
    output_func=ens_output_func,
    prob_func=prob_func)

sim = solve(ens_prob, ode_alg, trajectories=num_traj)
# make a vector of matrix into a 3D array

extract = [sim.u[i][j, k] for i = 1:num_traj, j = 1:2, k = 1:301]

max_arr = [maximum(extract[:, j, k]) for j = 1:2, k = 1:301]
min_arr = [minimum(extract[:, j, k]) for j = 1:2, k = 1:301]
mean_arr = [mean(extract[:, j, k]) for j = 1:2, k = 1:301]
p1 = plot(0:0.1:10, max_arr[1, 1:101], fillrange=(min_arr[1, 1:101]), alpha=0.5, label=L"range(x_1)", color=:green, xlabel=L"t", ylabel=L"x_1", legend=:outertopright, legendfontsize=12);
plot!(0:0.1:10, min_arr[1, 1:101], label="", color=:green, alpha=0.5);
plot!(p1, 0:0.1:10, mean_arr[1, 1:101], label=L"x_1", color=:green, width=3);
plot!(p1, 0:0.1:10, ones(101) * 0.38952, label=L"x_{1,nom}", color=:black, width=1);

p2 = plot(0:0.1:10, max_arr[2, 1:101], fillrange=(min_arr[2, 1:101]), alpha=0.5, label=L"range(x_2)", legendfontsize=12,
    color=:red, xlabel=L"t", ylabel=L"x_2", legend=:outertopright);
plot!(0:0.1:10, min_arr[2, 1:101], label="", color=:red, alpha=0.5);

plot!(p2, 0:0.1:10, mean_arr[2, 1:101], label=L"x_2", color=:red, width=3);
plot!(p2, 0:0.1:10, ones(101) * 0.606904, label=L"x_{2,nom}", color=:black, width=1);
ylims!(p1, (0.0, 1.0))
ylims!(p2, (0.0, 1.0))

p3 = plot(p1, p2, layout=(2, 1), label="")

savefig(p3, plotsdir("Compressor/compressor_" * case * ".svg"))

using DiffEqUncertainty, IntegralsCubature
# calculate the Expected closed loop cost of the mpc policy at different points
function V_cl(sol)
    return sol[3, end]
end
# IntegralsCubature.CubatureJLp() 0.1

function E_Vcl(prob, x1, x2)
    e = expectation(V_cl, prob, [x1; x2; 0.e0; 0.e0], [p_dis; 0.0], Koopman(), ode_alg;
        quadalg=IntegralsCubature.CubatureJLp(), iabstol=1e-2, ireltol=1e-1)[1]
    return e
end
@time E_Vcl(mpc_prob, x0[1], x0[2])
@time E_Vcl(nn_prob, x0[1], x0[2])

levels = collect(range(0.0, 110.0, length=10))
x1s = collect(range(0.2, 1.0, length=10))
x2s = collect(range(0.0, 0.8, length=10))

E_Vnn = zeros(length(x2s), length(x1s))
E_Vmpc = zeros(length(x2s), length(x1s))

for i = 1:length(x1s)
    for j = 1:length(x2s)
        println(j + (i - 1) * length(x2s))
        E_Vnn[j, i] = E_Vcl(nn_prob, x1s[i], x2s[j]) # the weird indexing is correct.
    end
end
for i = 1:length(x1s)
    for j = 1:length(x2s)
        # print the number of loops
        println(j + (i - 1) * length(x2s))
        E_Vmpc[j, i] = E_Vcl(mpc_prob, x1s[i], x2s[j]) # the weird indexing is correct.
    end
end
E_Vmpc[E_Vmpc.>300] .= Inf

tv = -1:0.5:2
tl = [L"10^{%$i}" for i in tv]
using Plots.Measures
Vnn_contf = contour(x1s, x2s, log10.(E_Vnn), fill=true, color=:viridis, colorbartitle=" \n" * L"\log_{10}\ V_{cl}", colorbar_titlefontsize=14, rightmargin=10mm)
savefig(Vnn_contf, plotsdir("compr_EV_nn.svg"))

for c_x = 1:3
    for c_p = 1:2
        if c_p == 2
            sty = (color=:red, style=:dash,)
        elseif c_p == 3
            sty = (color=:orange, style=:dashdot,)
        else
            sty = (color=:pink,)
        end
        nn_prob = ODEProblem(nn_system!, pl_x0[:, c_x], plot_tf, pl_ppp[:, c_p])
        sol = solve(nn_prob, ode_alg)
        plot!(Vnn_contf, sol(0)[1, :], sol(0)[2, :], label="", linewidth=2, marker=:circle, markersize=5; sty...)
        plot!(Vnn_contf, sol(0:0.1:plot_tf)[1, :], sol(0:0.1:plot_tf)[2, :], label="", linewidth=2; sty...)
    end
end
xlims!(Vnn_contf, (0.2, 1.0))
ylims!(Vnn_contf, (0.0, 0.8))
savefig(Vnn_contf, plotsdir("compr_EV_nn_traj.svg"))

# may need to adjust colorbarfontsize here.
mpc_contf = contourf(x1s, x2s, log10.(E_Vmpc), color=:viridis,
    colorbar_title=" \n" * L"\log_{10} E_{Vmpc}", colorbar_titlefontsize=14, rightmargin=10mm)
savefig(mpc_contf, plotsdir("compr_EV_mpc.svg"))

dif_E = (E_Vmpc .- E_Vnn) ./ E_Vnn
diff_contf = contourf(x1s, x2s, 100 * dif_E, color=:viridis, levels=11, colorbar_title=" \n" * L"100\% (E_{Vmpc} - E_{Vnn})/E_{Vnn}", colorbar_titlefontsize=14, rightmargin=10mm)
savefig(diff_contf, plotsdir("compr_EV_diff.svg"))

# save E_Vmpc

save_params = @strdict E_Vmpc E_Vnn x1s x2s
file_name = datadir("compressor/E_V_data") * ".jld2"
save(file_name, save_params)

