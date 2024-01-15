addpath '/home/evren/MATLAB/tbxmanager'
tbxmanager restorepath

% https://www.mpt3.org/
nx=6;
nu = 2;
% sys = drss(nx,nx,nu);

load('sys.mat');
load('expmpc.mat');
% writematrix(sys.A,'Amat.csv')
% writematrix(sys.b,'bmat.csv')
% writematrix(expmpc.model.x.terminalPenalty.weight, 'Pmat.csv')
plotoptions = pzoptions;
plotoptions.Grid = 'on';
plotoptions.GridColor = [0,0,0];
h = pzplot(sys, 'r', plotoptions);

model = LTISystem('A', sys.A, 'B', sys.B);


model.x.min = -1*ones(nx,1);
model.x.max = ones(nx,1);
model.u.min = -1*ones(nu,1);
model.u.max = ones(nu,1);


Q = eye(nx);
model.x.penalty = QuadFunction(Q);

R = 0.5*eye(nu);
model.u.penalty = QuadFunction(R);

PN = model.LQRPenalty;
model.x.with('terminalPenalty');
model.x.terminalPenalty = PN;


ini_set = Polyhedron('lb', ones(nx,1)*-0.44, 'ub', ones(nx,1)*0.44)


% but where is reachable from the ini_set?
R =model.reachableSet('X',ini_set,'N',1, 'direction', 'forward')
% we need to define the explicit control law on this set otherwise cannot
% be used in CL.

% model.x.with('initialSet');
% model.x.initialSet = R;
lqr_model = model;
model.x.min = -inf*ones(nx,1);
model.x.max = inf*ones(nx,1);
model.u.min = -inf*ones(nu,1);
model.u.max = inf*ones(nu,1);

N = 8;
mpc = MPCController(model, N);
lqr = MPCController(lqr_model, N);
% expmpc = mpc.toExplicit() % 6505 regions
x0 =0.44*ones(nx,1);

f = @() expmpc.evaluate(0.1*ones(nx,1))
timeit(f)
% approx 0.0158 seconds


loop = ClosedLoop(lqr, lqr_model);

loop = ClosedLoop(expmpc, model);
data = loop.simulate(x0, N);




ctrl_simple = mpt_simplify(expmpc)
ctrl_inv = mpt_invariantSet(expmpc)

expmpc.clicksim()
%expmpc.cost.fplot()
%expmpc.feedback.fplot()