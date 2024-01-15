"""
Provides a function to solve double integrator problem given an initial guess.
"""
function solve_mpc(x0)
	Ad = Float64[1. 1.; 0. 1.]
	Bd = Float64[0.; 1.]
	P = [2.177565296080891 1.2571126613074837; 1.2571126613074837 1.480332243219584]


	N_horizon = 4

	# Define cost function weights
	Q = [1.0 0.0; 0.0 0.05] # state cost
	R = 0.1 # input cost

	mpc= Model(optimizer_with_attributes(Ipopt.Optimizer,
	"linear_solver" => "ma97", "print_level" => 0))
	
	@variable(mpc, x[1:2, 1:N_horizon+1])
	@variable(mpc, -3 <= u[1:N_horizon] <=3)


	# initial state constraint
	@constraint(mpc, x[:, 1] .== x0)

	# dynamics constraints
	for k in 1:N_horizon
		@constraint(mpc, x[:, k+1] .== Ad*x[:, k] + Bd*u[k])
	end

	@expression(mpc, cost_running, sum(x[:, k]'*Q*x[:, k] + u[k]'*R*u[k] for k in 1:N_horizon))
	@expression(mpc, cost_final, x[:, end]'*P*x[:, end] )

	
	for k in 1:N_horizon+1
		set_lower_bound(x[2,k], -3)
		set_upper_bound(x[2,k], 3)
	end
	
	@objective(mpc, Min, cost_running + cost_final)

	#=
	# create slack variables for x[2] constraint
	@variable(mpc, 0 <= x2_slack[2,1:N_horizon+1])
	for k in 1:N_horizon+1
		@constraint(mpc, x[2,k] <= 3 + x2_slack[1,k])
		@constraint(mpc, x[2,k] >= -3 - x2_slack[2,k])
	end
	@expression(mpc, penal_cost, 10*sum(x2_slack[1,k] + x2_slack[2,k] for k in 1:N_horizon+1))
	@objective(mpc, Min, cost_running + cost_final + penal_cost)
	=#



	JuMP.optimize!(mpc)


return value.(u)[1]
end

function solve_open_loop(x0)
	Ad = Float64[1. 1.; 0. 1.]
	Bd = Float64[0.; 1.]
	P = [2.177565296080891 1.2571126613074837; 1.2571126613074837 1.480332243219584]


	N_horizon = 4

	# Define cost function weights
	Q = [1.0 0.0; 0.0 0.05] # state cost
	R = 0.1 # input cost

	mpc= Model(optimizer_with_attributes(Ipopt.Optimizer,
	"linear_solver" => "ma97", "print_level" => 0))
	
	@variable(mpc, x[1:2, 1:N_horizon+1])
	@variable(mpc, -3 <= u[1:N_horizon] <=3)


	# initial state constraint
	@constraint(mpc, x[:, 1] .== x0)

	# dynamics constraints
	for k in 1:N_horizon
		@constraint(mpc, x[:, k+1] .== Ad*x[:, k] + Bd*u[k])
	end

	@expression(mpc, cost_running, sum(x[:, k]'*Q*x[:, k] + u[k]'*R*u[k] for k in 1:N_horizon))
	@expression(mpc, cost_final, x[:, end]'*P*x[:, end] )

	
	for k in 1:N_horizon+1
		set_lower_bound(x[2,k], -3)
		set_upper_bound(x[2,k], 3)
	end
	
	@objective(mpc, Min, cost_running + cost_final)

	#=
	# create slack variables for x[2] constraint
	@variable(mpc, 0 <= x2_slack[2,1:N_horizon+1])
	for k in 1:N_horizon+1
		@constraint(mpc, x[2,k] <= 3 + x2_slack[1,k])
		@constraint(mpc, x[2,k] >= -3 - x2_slack[2,k])
	end
	@expression(mpc, penal_cost, 10*sum(x2_slack[1,k] + x2_slack[2,k] for k in 1:N_horizon+1))
	@objective(mpc, Min, cost_running + cost_final + penal_cost)
	=#



	JuMP.optimize!(mpc)


return value.(u)
end