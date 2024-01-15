"""
Provides a function to solve compressor problem given an initial guess.
"""
# x0 = [.1 375]
function solve_mpc(x0)
	B=1
	H=0.18
	ψc = 0.35
	W=0.25
	_γ=0.5
	h =0.5
	xs = [0.4;0.6]
	N_horizon = 12

	mpc= Model(optimizer_with_attributes(Ipopt.Optimizer,
	"linear_solver" => "ma97", "print_level" => 0))
	
	@variable(mpc, x[1:2,1:N_horizon+1], start=0.5)
	@variable(mpc, 0. <= u[1:N_horizon] <=0.3, start=0.1)

	# introduce slack variables for state constraints
	@variable(mpc, 0 <= s_low[1:2, 1:N_horizon+1], start=0.0)
	@variable(mpc, 0 <= s_high[1:2, 1:N_horizon+1], start=0.0)

	for j=1:2
		for k=1:N_horizon+1
			@constraint(mpc, x[j,k] <= 1.0 + s_high[j,k])
			@constraint(mpc, x[j,k] >= 0.0 - s_low[j,k])
		end
	end


	@NLexpression(mpc, ψe[k=1:N_horizon+1], ψc + H*(1+1.5*(x[1,k]/W -1) - 0.5*(x[1,k]/W -1)^3))

	@NLexpression(mpc, ϕ[k=1:N_horizon+1], _γ*sqrt(x[2,k]))

	@variable(mpc, 0 <= v[1:N_horizon+1], start=0.0)

	@constraint(mpc, x[:,1] .== x0)

	for k in 1:N_horizon
		@NLconstraint(mpc, x[1,k+1] == x[1,k] + h*(B*(ψe[k+1]-x[2,k+1]-u[k])))
		@NLconstraint(mpc, x[2,k+1] == x[2,k] + h*((1/B)*(x[1,k+1]-ϕ[k+1])))
		@constraint(mpc, x[2,k] ≥ 0.4 - v[k])
	end

	@objective(mpc, Min, 100*sum((x[:,k] .-xs)'*(x[:,k].-xs) for k=1:N_horizon+1) + 8*u'*u + 800*v'*v + 
	1000*sum(s_low[j,k] +s_high[j,k] for k=1:N_horizon+1, j=1:2
	))
	JuMP.optimize!(mpc)


	if termination_status(mpc) == LOCALLY_SOLVED || termination_status(mpc) == OPTIMAL
		return value.(u)[1], objective_value(mpc) 	else
		return Inf, Inf
	end

end