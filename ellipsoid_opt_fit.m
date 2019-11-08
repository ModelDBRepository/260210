function [W fval] = ellipsoid_opt_fit(xdata)
obFunc = @(W) cost_fn(xdata, W);
W0 = 2*rand(6,1);
LB = 0*ones(6,1);
UB = 20*ones(6,1);
nonlcon = @unit_vec_con;
[W, fval] = fmincon(obFunc, W0, [], [], [], [], LB, UB);

