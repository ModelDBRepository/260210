function bs = bs_compute(firingmap2)
sigma=5;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0;
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90); 
bs_arr_XY = [];
for ii = 1:size(b2,2)
    a = squeeze(b2(:,:,ii)); a(isnan(a)) = 0;
    c.map = (a>0);
    bs_arr_XY(ii) = borderScore(a,a,c);
end

%
bs_arr_YZ = [];
for ii = 1:size(b2,2)
    a = squeeze(b2(ii,:,:)); a(isnan(a)) = 0;
    c.map = (a>0);
    bs_arr_YZ(ii) = borderScore(a,a,c);
end

%
bs_arr_XZ = [];
for ii = 1:size(b2,2)
    a = squeeze(b2(:,ii,:)); a(isnan(a)) = 0;
    c.map = (a>0);
    bs_arr_XZ(ii) = borderScore(a,a,c);
end

bs_arr = sort([nanmean(bs_arr_XY) nanmean(bs_arr_YZ) nanmean(bs_arr_XZ)]);
bs = max([bs_arr(end) bs_arr(end-1)]);
end