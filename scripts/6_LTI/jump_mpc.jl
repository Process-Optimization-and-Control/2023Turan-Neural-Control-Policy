using DrWatson 
@quickactivate "nn_controller"
using JuMP, Ipopt
include(scriptsdir("6_LTI","par.jl"))
		
"""
Provides a function to solve mpc problem given an initial guess.
"""
function solve_mpc(x0)

	mpc= Model(optimizer_with_attributes(Ipopt.Optimizer,
	"linear_solver" => "ma97", "print_level" => 0))
	
	@variable(mpc,  -1<= x[1:nx, 1:N_horizon+1]<=1, start = 0.5)
	@variable(mpc, -1 <= u[1:nu, 1:N_horizon] <=1, start = 0.5)


	# initial state constraint
	@constraint(mpc, x[:, 1] .== x0)

	# dynamics constraints
	for k in 1:N_horizon
		@constraint(mpc, x[:, k+1] .== Ad*x[:, k] + Bd*u[:,k])
	end

	@expression(mpc, cost_running, sum(x[:, k]'*Q*x[:, k] + u[:,k]'*R*u[:,k] for k in 1:N_horizon))
	@expression(mpc, cost_final, x[:, end]'*P*x[:, end] )


	@objective(mpc, Min, cost_running + cost_final)

	JuMP.optimize!(mpc)


return value.(u)[:,1]
end
