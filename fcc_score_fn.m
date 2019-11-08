function fcc_score = fcc_score_fn(c)
%% FCC analytical
a=0.5;
res = 25;
%FCC grid
K = (2*pi/a)*[0 0 sqrt(3/2); 2/sqrt(3) 0 -1/sqrt(6); -1/sqrt(3) 1 -1/sqrt(6)];
[k1,k2,k3] = meshgrid(1:1/res:3);
pos2=[reshape(k1,prod(size(k1)),1) reshape(k2,prod(size(k1)),1) reshape(k3,prod(size(k1)),1)]';
arg = K*pos2;
R = (1/3)*sum(cos(arg))+1;
R2=reshape(R,ceil(size(R,2)^(1/3)),ceil(size(R,2)^(1/3)),ceil(size(R,2)^(1/3)));
R2_wo_nan = R2;
%% Gridness score computation
[mean_gscore_analytical gscore_arr_analytical] = fcc_gscore(R2_wo_nan);
[mean_gscore_network gscore_arr_network] = fcc_gscore(c);
fcc_score = ((mean_gscore_network>0)*mean_gscore_network) / mean_gscore_analytical;
% fcc_score = (mean_gscore_network) / mean_gscore_analytical;
end
