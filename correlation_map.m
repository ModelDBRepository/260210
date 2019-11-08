function Rxy = correlation_map(map1,map2)

% map1(isnan(map2)) = NaN;
% map2(isnan(map1)) = NaN;

bins = size(map1,1);
N = bins + round(0.8*bins);
if ~mod(N,2)
    N = N - 1;
end
% Centre bin
cb = (N+1)/2;
Rxy = zeros(N);
for ii = 1:N
    rowOff = ii-cb;
    for jj = 1:N
        colOff = jj-cb;
        Rxy(ii,jj) = pointCorr(map1,map2,rowOff,colOff,bins);
    end
end