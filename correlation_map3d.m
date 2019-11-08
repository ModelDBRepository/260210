function Rxyz = correlation_map3d(map1,map2)

map1(isnan(map2)) = NaN;
map2(isnan(map1)) = NaN;

bins = size(map1,1);
N = bins  + round(0.9*bins);
if ~mod(N,2)
    N = N - 1;
end
% Centre bin
cb = (N+1)/2;
Rxyz = zeros(N,N,N);
for ii = 1:N
    rowOff = ii-cb;
    for jj = 1:N
        colOff = jj-cb;
        for kk = 1:N
            hcolOff = kk-cb;
            Rxyz(ii,jj,kk) = pointCorr3d(map1,map2,rowOff,colOff,hcolOff,bins);
        end
    end
end
end