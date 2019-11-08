%This code shows examples of spatial representations emerging from the heirarchichal neural network model of 3d space. 
%For more details on the heirarchical layers of the model see the code named "model_code".
clc
clear all
close all
% Load trajectory
load('trj_1')
% HD estimation
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
dendphase_az = [];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+(2*pi*dendfreq(ii,:)*dt);
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=cos(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=cos(dendphase);
    baseosc=cos(basephase);
    piosc_az(:,ii)=dendosc';
    dendphase_az(:,ii) = dendphase;
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1);
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
dendphase_pitch = [];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=cos(dendphase);
    piosc_pitch(:,ii)=dendosc';
    dendphase_pitch(:,ii)=dendphase';
end
piosc_tot = [piosc_az ;piosc_pitch];
dendphase_tot = [dendphase_az ;dendphase_pitch];
piosc_thresh_tot=(abs(piosc_tot)>0.9).*piosc_tot;
% Compute occupancy
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
osccupancy_time_bin_flag = nan(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
osccupancy_time_bin_flag = fliplr(osccupancy_time_bin_flag);
% Place 1
load('lahn_wt_1')
neuron_nmbr = 9;
w = T(neuron_nmbr,:)';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res_fr = 20;
[fx,fy,fz] = meshgrid(1:1/res_fr:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:size(firposgrid,1)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.15*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-50 34])
title('Place 1')
% Place 2
load('lahn_wt_2')
piosc_thresh_tot=(abs(piosc_tot)>0.9).*piosc_tot;
neuron_nmbr = 23;
w = T(neuron_nmbr,:)';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res_fr = 20;
[fx,fy,fz] = meshgrid(1:1/res_fr:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:size(firposgrid,1)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.2*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-50 34])
title('Place 2')
% Place 3
load('lahn_wt_3')
piosc_thresh_tot=(abs(piosc_tot)>0.9).*piosc_tot;
neuron_nmbr = 23;
w = T(neuron_nmbr,:)';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res_fr = 20;
[fx,fy,fz] = meshgrid(1:1/res_fr:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:size(firposgrid,1)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.2*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90); 
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-50 34])
title('Place 3')
% Place 4
load('lahn_wt_4')
piosc_thresh_tot=(abs(piosc_tot)>0.9).*piosc_tot;
neuron_nmbr = 12;
w = T(neuron_nmbr,:)';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res_fr = 20;
[fx,fy,fz] = meshgrid(1:1/res_fr:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:size(firposgrid,1)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.2*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90); 
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-50 34])
title('Place 4')
train_flag = 0;
if train_flag
    PI1d = piosc_thresh_tot;
    [N K] = size(PI1d); %N --> Dimension    K---> # of samples
    PI1d=removemean(PI1d);
    alphaa = 0.0001/K;
    betaa = 0.00001/K;
    output_neuron_nmbr = 40;
    maxiter = 2000000;
    [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);
    SI = [];
    b2_arr = [];
    for neuron_index = 1:output_neuron_nmbr
        neuron_index
        neuron_nmbr = neuron_index;
        w = T(neuron_nmbr,:)';
        ot=w'*piosc_thresh_tot; ot=ot';
        ot=abs(ot);
        thresh=max(ot)*.75;
        firr=[];
        firr=find((ot)>thresh);
        firposgrid=pos(firr,:);
        
        [fx,fy,fz] = meshgrid(1:1/res:3);
        firingmap = zeros(length(fx),length(fx),length(fx));
        gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
        roundinggridpoint = round(gridpoint);
        firposround = round(firposgrid);
        firingvalue = (ot(firr));
        for ii = 1:size(firposgrid,1)
            [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
            [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
            [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
            firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
        end
        firingmap2 = firingmap./(osccupancy_time_bin+eps);
        firingmap2=fliplr(firingmap);
        
        sigma=3;
        b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
        thresh=0.2*max(max(max((b))));
        b(abs(b)<thresh)=nan;
        b2=imrotate(b/max(max(max(b))),90);
        
        qq = isnan(osccupancy_time_bin_flag(:)); q = find(qq==1);
        spikes_smooth_1d = spikes_smooth(:);
        spikes_smooth_1d(q)=nan;
        Nbins = size(spikes_smooth,1);
        p = fliplr(osccupancy_time_bin)/size(pos,1);
        fr_mean = nanmean(spikes_smooth_1d);
        fr_nor = (spikes_smooth+eps)/fr_mean;
        fr_log = log2((spikes_smooth+eps)/fr_mean);
        SI(neuron_index) = sum(sum(sum(p.*fr_nor.*fr_log)));
        b2_arr(:,:,:,neuron_index) = b2;
    end    
    q =  find(SI> 2.4);
    figure
    for ii = 1:length(q)
        subplot(1,length(q),ii)
        b2 = b2_arr(:,:,:,q(ii));
        h = slice(b2, [], [], 1:size(b,3));
        set(h, 'EdgeColor','none', 'FaceColor','interp')
        alpha(.8)
        colormap(jet)
        axis off
        view([-56 19])
        hold on
        X = [1;size(b,1);size(b,1);1;1];
        Y = [1;1;size(b,2);size(b,2);1];
        Z = [1;1;1;1;1];
        minpos=1; maxpos=3;
        pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
        plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
        plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
        plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
        for k=1:length(X)-1
            plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
        end
        view([-35 21])
    end
end
%Elongation index plot
load('elongation_index')
figure; histfit(E,11);
xlim([1 2.01])
ylim([0 40])
title('Elongation index')
ylabel('Count'); xlabel('Elongation index')
train_flag = 0;
if train_flag
    niter = 100;
    count = 0;
    elongation_index = [];
    for itr_ind = 1:niter
        itr_ind
        [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);
        SI = [];
        for neuron_index = 1:output_neuron_nmbr
            neuron_index
            neuron_nmbr = neuron_index;
            w = T(neuron_nmbr,:)';
            ot=w'*piosc_thresh_tot; ot=ot';
            ot=abs(ot);
            thresh=max(ot)*.75;
            firr=[];
            firr=find((ot)>thresh);
            firposgrid=pos(firr,:);
            
            [fx,fy,fz] = meshgrid(1:1/res:3);
            firingmap = zeros(length(fx),length(fx),length(fx));
            gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
            roundinggridpoint = round(gridpoint);
            firposround = round(firposgrid);
            firingvalue = (ot(firr));
            for ii = 1:size(firposgrid,1)
                [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
                [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
                [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
                firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
            end
            firingmap2 = firingmap./(osccupancy_time_bin+eps);
            firingmap2=fliplr(firingmap);
            
            sigma=3;
            b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
            thresh=0.15*max(max(max((b))));
            b(abs(b)<thresh)=nan;
            b2=imrotate(b/max(max(max(b))),90);
            
            qq = isnan(osccupancy_time_bin_flag(:)); q = find(qq==1);
            spikes_smooth_1d = spikes_smooth(:);
            spikes_smooth_1d(q)=nan;
            Nbins = size(spikes_smooth,1);
            p = fliplr(osccupancy_time_bin)/size(pos,1);
            fr_mean = nanmean(spikes_smooth_1d);
            fr_nor = (spikes_smooth+eps)/fr_mean;
            fr_log = log2((spikes_smooth+eps)/fr_mean);
            SI(neuron_index) = sum(sum(sum(p.*fr_nor.*fr_log)));
        end
        q = find(SI>2.5)
        for plc_ind = 1:length(q)
            sel = plc_ind;
            w = T(q(sel),:)';
            ot=w'*piosc_thresh_tot; ot=ot';
            ot=abs(ot);
            thresh=max(ot)*.75;
            firr=[];
            firr=find((ot)>thresh);
            firposgrid=pos(firr,:);
            
            [fx,fy,fz] = meshgrid(1:1/res:3);
            firingmap = zeros(length(fx),length(fx),length(fx));
            gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
            roundinggridpoint = round(gridpoint);
            firposround = round(firposgrid);
            firingvalue = (ot(firr));
            for ii = 1:size(firposgrid,1)
                [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
                [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
                [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
                firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
            end
            firingmap2 = firingmap./(osccupancy_time_bin+eps);
            firingmap2=fliplr(firingmap);
            
            sigma=3;
            b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
            thresh=0.22*max(max(max((b))));
            b(abs(b)<thresh)=nan;
            b2=imrotate(b/max(max(max(b))),90);
            rate_map_3D_thresh = 0.5*max(max(max(spikes_smooth)));
            rate_map_3D = spikes_smooth;
            rate_map_3D_compare=(rate_map_3D>rate_map_3D_thresh);
            rate_map_3D_compare=logical(rate_map_3D_compare);
            STATS = regionprops(rate_map_3D_compare,'PixelIdxList');
            voxel_ind = STATS.PixelIdxList;
            single_field = unique(voxel_ind);
            [x,y,z] = ind2sub(size(rate_map_3D),single_field);
            X = [x,y,z];
            [W fval] = ellipsoid_opt_fit(X);
            axis_length = sqrt([W(4) W(5) W(6)]);
            count = count + 1;
            elongation_index(count) = max(axis_length)/min(axis_length)';
        end
    end
    figure; histfit(elongation_index,11);
    ylabel('Count'); xlabel('Elongation index')
    xlim([1 3])
end
% Load a trajectory
load('trj_2')
% HD estimation of the trajectory
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Trj stati
th = -180:180;
pitch_angle = theta_pitch;
[counts, binValues] = hist(pitch_angle,th);
normalizedCounts = counts / max(counts);
[p,q]=max(normalizedCounts);
normalizedCounts = normalizedCounts / max(normalizedCounts);
figure; bar(binValues, normalizedCounts, 'barwidth', 1);
xlim([-90 90])
xlabel('Pitch angle in degree');
ylabel('Normalized Frequency');
title('Pitch distribution')
az_angle = theta_az;
[counts, binValues] = hist(az_angle,th);
normalizedCounts = counts / max(counts);
[p,q]=max(normalizedCounts);
normalizedCounts = normalizedCounts / max(normalizedCounts);
figure; bar(binValues, normalizedCounts, 'barwidth', 1);
xlim([-180 180])
xlabel('Azimuth angle in degree');
ylabel('Normalized Frequency');
title('Azimuth distribution')
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=sin(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=sin(dendphase);
    baseosc=sin(basephase);
    piosc_az(:,ii)=dendosc';
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1);
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=sin(dendphase);
    piosc_pitch(:,ii)=dendosc';
end
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=((piosc_tot)>0.9).*piosc_tot;
% Occupancy of the trajectory
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
osccupancy_time_bin_flag = nan(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
osccupancy_time_bin_flag = fliplr(osccupancy_time_bin_flag);
% Grid 1
load('lahn_wt_5');
w=T(20,:);w = w';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
d1=1; d2=2;
firposgrid2=firposgrid(:,d1:d2);
% 3D rate map;
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap;
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.31*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.45)
colormap(jet)
axis off
view([-53 31])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
title('Grid 1')
piosc_thresh_tot=((piosc_tot)>0.5).*piosc_tot;
w=T(21,:);w = w'; ot=w'*piosc_thresh_tot; ot=ot'; ot=abs(ot);
thresh=max(ot)*.75;
firr=[]; firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res = 30;
d1 = 1; d2 = 2;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
cen = size(firingmap,1)/2;
gaussian = fspecial('gaussian',[10 10],15);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
figure; imagesc(Rxy)
colormap(jet)
axis off
title('Hexagonal autocorrelation map')
% Grid 2
fr_hist=[];
load('lahn_wt_6')
w=T(33,:);w = w';
piosc_thresh_tot=((piosc_tot)>0.9).*piosc_tot;
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
d1=1; d2=2;
firposgrid2=firposgrid(:,d1:d2);
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.3*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.45)
colormap(jet)
axis off
view([-68 32])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
title('Grid 2')
w=T(5,:);w = w'; ot=w'*piosc_thresh_tot; ot=ot'; ot=abs(ot);
thresh=max(ot)*.8;
firr=[]; firr=find((ot)>thresh);
firposgrid=pos(firr,:);
res = 23;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
gaussian = fspecial('gaussian',10*[1 1],10);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
figure; imagesc(Rxy)
colormap(jet)
axis off
title('Square autocorrelation map')
train_flag = 0;
if train_flag
    PI1d = piosc_thresh_tot;
    [N K] = size(PI1d); %N --> Dimension    K---> # of samples
    PI1d=removemean(PI1d);
    alphaa = 0.0001/K;
    betaa = 0.00001/K;
    output_neuron_nmbr = 40;
    maxiter = 2000000;
    [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);
    hgs = []; sgs = [];
    Rxy_arr = [];
    res = 20;
    [fx,fy] = meshgrid(1:1/res:3);
    osccupancy_time_bin_2d = zeros(size(fx));
    for ii = 1:size(pos,1)
        ii
        [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
        [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
        osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
    end
    for neuron_index = 1:output_neuron_nmbr
        neuron_index
        neuron_nmbr = neuron_index;
        w = T(neuron_nmbr,:)';
        ot=w'*piosc_thresh_tot; ot=ot';
        ot=abs(ot);
        thresh=max(ot)*.75;
        firr=[];
        firr=find((ot)>thresh);
        firposgrid=pos(firr,:);
        d1=1; d2=2;
        firposgrid2=firposgrid(:,d1:d2);
        firingmap = zeros(length(fx));
        gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
        roundinggridpoint = round(gridpoint);
        firposround = round(firposgrid2);
        firingvalue = abs(ot(firr));
        for ii = 1:length(firposgrid2)
            [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
            [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
            firingmap(q1,q2) = firingmap(q1,q2) + 1;
        end
        firingmap = firingmap./(osccupancy_time_bin_2d+eps);
        gaussian = fspecial('gaussian',10*[1 1],10);
        spikes_smooth=conv2(gaussian,firingmap);
        spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
        Rxy = correlation_map(spikes_smooth,spikes_smooth);
        Rxy_arr(:,:,neuron_index) = Rxy;
        hgs(neuron_index) = gridscore(Rxy);
        sgs(neuron_index) = squaregridscore(Rxy);
    end
    q1 = find(hgs>0.1);
    q2 = find(sgs>0.2);
    q = [q1 q2];
    figure
    for ii = 1:length(q)
        Rxy = Rxy_arr(:,:,q(ii));
        subplot(5,8,ii)
        imagesc(Rxy)
        colormap(jet)
        axis off
        axis equal
    end
end
%FCC score
load('FCC score of grid neurons')
numOfBins = 10;
[histFreq, histXout] = hist(E, numOfBins);
figure;
histfit(E,numOfBins); hold on
xlim([0 1])
title('FCC score distribution')
train_flag = 0;
if train_flag
    PI1d = piosc_thresh_tot;
    [N K] = size(PI1d); %N --> Dimension    K---> # of samples
    PI1d=removemean(PI1d);
    alphaa = 0.0001/K;
    betaa = 0.00001/K;
    niter = 50;
    output_neuron_nmbr = 40;
    maxiter = 2000000;
    [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);
    count = 0;
    fcc_val_arr = [];
    for itr_ind = 1:niter
        for neuron_index = 1:output_neuron_nmbr
            neuron_index
            w = T(neuron_index,:)';
            ot=w'*piosc_thresh_tot; ot=ot';
            ot=abs(ot);
            thresh=max(ot)*.75;
            firr=[];
            firr=find((ot)>thresh);
            firposgrid=pos(firr,:);
            res = 20;
            [fx,fy,fz] = meshgrid(1:1/res:3);
            firingmap = zeros(length(fx),length(fx),length(fx));
            gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
            roundinggridpoint = round(gridpoint);
            firposround = round(firposgrid);
            firingvalue = abs(ot(firr));
            for ii = 1:length(firposgrid)
                [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
                [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
                [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
                firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
            end
            firingmap2 = firingmap./(osccupancy_time_bin+eps);
            sigma=3;
            b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
            thresh=0.1*max(max(max((b))));
            b(abs(b)<thresh)=nan;
            b2=imrotate(b/max(max(max(b))),90);
            Rxyz = correlation_map3d(spikes_smooth,spikes_smooth);
            c = Rxyz;
            thresh=0.2*max(max(max(c)));
            c(c<thresh)=nan;
            fcc_val = fcc_score_fn(c);
            if fcc_val~=0
                count = count + 1;
                fcc_val_arr(count) = fcc_val;
            end
        end
    end
end
% Isotropy checking
load('trj_2')
pos(:,3) = 1; %On XY plane
% HD estimation
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=sin(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=sin(dendphase);
    baseosc=sin(basephase);
    piosc_az(:,ii)=dendosc';
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1);
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=sin(dendphase);
    piosc_pitch(:,ii)=dendosc';
end
piosc_tot = [piosc_az ;piosc_pitch];
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
osccupancy_time_bin_flag = nan(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
osccupancy_time_bin_flag = fliplr(osccupancy_time_bin_flag);
piosc_thresh_tot=(abs(piosc_tot)>0.9);
%Place isotropy on XY plane
load('lahn_wt_7')
w=T(6,:);w=w';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.8;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); % set the azimuth and elevation of the plot [-28 35]
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Place field on XY plane')
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.15*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1);
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
% plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-45 34])
title('Place field on XY plane')
%Grid isotropy on XY plane
load('lahn_wt_8');
w=T(21,:);w = w'; 
piosc_thresh_tot=(abs(piosc_tot)>0.5);
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.8;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); % set the azimuth and elevation of the plot [-28 35]
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Grid field on XY plane')
d1 = 1; d2 = 2;
res_2d = 40;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res_2d:3);
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
cen = size(firingmap,1)/2;
gaussian = fspecial('gaussian',[10 10],15);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
figure; imagesc(Rxy)
colormap(jet)
axis off
title('Grid autocorrelation map on XY plane')
% Anisotropy checking
load('trj_2')
pos(:,1) = 1; %On YZ plane
% HD estimation
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=sin(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=sin(dendphase);
    baseosc=sin(basephase);
    piosc_az(:,ii)=dendosc';
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1);
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=sin(dendphase);
    piosc_pitch(:,ii)=dendosc';
end
piosc_tot = [piosc_az ;piosc_pitch];
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
osccupancy_time_bin_flag = nan(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
osccupancy_time_bin_flag = fliplr(osccupancy_time_bin_flag);
piosc_thresh_tot=(abs(piosc_tot)>0.9);
% Place field anisotropy
load('lahn_wt_7')
w=T(6,:);w=w';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.83;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); % set the azimuth and elevation of the plot [-28 35]
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Place field on YZ plane')
res = 50;
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
osccupancy_time_bin = zeros(size(fx));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
for ii = 1:size(pos,1)
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.015*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, 1, [], []);
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.8)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
% plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([-45 34])
title('Place field on YZ plane')
% Grid field anisotropy
load('lahn_wt_8');
w=T(21,:);w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.8;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); % set the azimuth and elevation of the plot [-28 35]
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Grid field on YZ plane')
d1 = 2; d2 = 3;
res_2d = 35;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res_2d:3);
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
cen = size(firingmap,1)/2;
gaussian = fspecial('gaussian',[10 10],15);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
figure; imagesc(Rxy)
colormap(jet)
axis off
title('Grid autocorrelation map on YZ plane')
% Isotropy with sd
%sd1
load('trj_2')
pos(:,1) = 1; %On YZ plane
% HD estimation
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=sin(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=sin(dendphase);
    baseosc=sin(basephase);
    piosc_az(:,ii)=dendosc';
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1);
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=sin(dendphase);
    piosc_pitch(:,ii)=dendosc';
end
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=(abs(piosc_tot)>0.9);
% Grid field anisotropy
load('lahn_wt_8');
w=T(21,:);w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
subplot(2,1,1)
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); 
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 15);
axis off
axis equal
title('Grid field on YZ plane')
d1 = 2; d2 = 3;
res_2d = 35;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res_2d:3);
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
cen = size(firingmap,1)/2;
gaussian = fspecial('gaussian',[10 10],15);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
subplot(2,1,2); imagesc(Rxy)
colormap(jet)
axis off
axis equal
title('Grid autocorrelation map on YZ plane')
%sd2
load('trj_1')
pos(:,1) = 1; %On YZ plane
% HD estimation
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2; %Spatial scaling parameter
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
dendphase_az = [];
X = zeros(n_Az,1); Y = ones(n_Az,1); 
for ii=1:length(pos)-1
    dendphase=dendphase+(2*pi*dendfreq(ii,:)*dt);
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=cos(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=cos(dendphase);
    baseosc=cos(basephase);
    piosc_az(:,ii)=dendosc';
    dendphase_az(:,ii) = dendphase;
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1); 
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
dendphase_pitch = [];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=cos(dendphase);
    piosc_pitch(:,ii)=dendosc';
    dendphase_pitch(:,ii)=dendphase';
end
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=(abs(piosc_tot)>0.9);
% Grid field anisotropy
piosc_thresh_tot=(abs(piosc_tot)>0.9);
load('lahn_wt_9');
w=T(21,:);w = w';
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
figure;
subplot(2,1,1)
X = 2*[0;1;1;0;0];
Y = 2*[0;0;1;1;0];
Z = 2*[0;0;0;0;0];
X = [1;3;3;1;1];
Y = [1;1;3;3;1];
Z = [1;1;1;1;1];
hold on;
plot3(X,Y,Z,'k');   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+2,'k'); % draw a square in the xy plane with z = 1
set(gca,'View',[28,15]); 
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k');
end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.5*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-45 34])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 15);
axis off
axis equal
title('Grid field on YZ plane')
d1 = 2; d2 = 3;
res_2d = 35;
firposgrid2=[firposgrid(:,d1) firposgrid(:,d2)];
[fx,fy] = meshgrid(1:1/res_2d:3);
osccupancy_time_bin_2d = zeros(size(fx));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-fx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-fx(1,:)));
    osccupancy_time_bin_2d(q1,q2) = osccupancy_time_bin_2d(q1,q2)+1;
end
firingmap = zeros(length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid2);
firingvalue = abs(ot(firr));
for ii = 1:length(firposgrid2)
    [~,q1]=min(abs(firposgrid2(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid2(ii,2)-fx(1,:)));
    firingmap(q1,q2) = firingmap(q1,q2) + 1;
end
firingmap = firingmap./(osccupancy_time_bin_2d+eps);
cen = size(firingmap,1)/2;
gaussian = fspecial('gaussian',[10 10],15);
spikes_smooth=conv2(gaussian,firingmap);
spikes_smooth=imrotate(spikes_smooth/max(max(spikes_smooth)),90);
Rxy = correlation_map(spikes_smooth,spikes_smooth);
subplot(2,1,2); imagesc(Rxy)
colormap(jet)
axis off
axis equal
title('Grid autocorrelation map on YZ plane')
% Plane field 1
load('trj_2')
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction; 
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch); 
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);   
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az'; 
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01; 
betaa=2; 
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
dendphase_az = [];
X = zeros(n_Az,1); Y = ones(n_Az,1); 
for ii=1:length(pos)-1
    dendphase=dendphase+(2*pi*dendfreq(ii,:)*dt); 
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=cos(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=cos(dendphase); 
    baseosc=cos(basephase); 
    piosc_az(:,ii)=dendosc';   
    dendphase_az(:,ii) = dendphase;
end
% PI_pitch layer
speed_pitch=s_pitch'; 
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1); 
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
dendphase_pitch = [];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt; 
    dendosc=cos(dendphase); 
    piosc_pitch(:,ii)=dendosc';  
    dendphase_pitch(:,ii)=dendphase';  
end
% Stacking piosc_Az and piosc_pitch
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=((piosc_tot)>0.9).*piosc_tot;
load('lahn_wt_10')
neuron_number = 4;
w=T(neuron_number,:);w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:); 
figure;
 X = [1;3;3;1;1];
 Y = [1;1;3;3;1];
 Z = [1;1;1;1;1];
 hold on;
 plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy e with z = 0
 plot3(X,Y,Z+2,'k','Linewidth',3); % draw a square in the xy plane with z = 1
 set(gca,'View',[32,25]); 
  for k=1:length(X)-1
     plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k','Linewidth',3);
  end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5); hold on;axis off
view([-175 17])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Plane field 1')
% Plane field 2
load('trj_3'); 
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction; 
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch); 
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);   
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az'; 
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01; 
betaa=2;  
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
dendphase_az = [];
X = zeros(n_Az,1); Y = ones(n_Az,1); 
for ii=1:length(pos)-1
    dendphase=dendphase+(2*pi*dendfreq(ii,:)*dt); 
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=cos(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=cos(dendphase); 
    baseosc=cos(basephase); 
    piosc_az(:,ii)=dendosc';   
    dendphase_az(:,ii) = dendphase;
end
% PI_pitch layer
speed_pitch=s_pitch'; 
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1); 
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
dendphase_pitch = [];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt; 
    dendosc=cos(dendphase); 
    piosc_pitch(:,ii)=dendosc';  
    dendphase_pitch(:,ii)=dendphase';  
end
% Stacking piosc_Az and piosc_pitch
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=((piosc_tot(:,1:49900))>0.9).*piosc_tot(:,1:49900);
neuron_number = 10;
w=T(neuron_number,:);w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.75;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:); 
figure;
 X = [1;3;3;1;1];
 Y = [1;1;3;3;1];
 Z = [1;1;1;1;1];
 hold on;
 plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy e with z = 0
 plot3(X,Y,Z+2,'k','Linewidth',3); % draw a square in the xy plane with z = 1
 set(gca,'View',[32,25]); 
  for k=1:length(X)-1
     plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k','Linewidth',3);
  end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5); hold on;axis off
view([81 18])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Plane field 2')
% Plane field 3
neuron_number = 11;
w=T(neuron_number,:);w = w'; %Selecting the weights of that neuron to remap from LAHN
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.78;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:); 
figure;
 X = [1;3;3;1;1];
 Y = [1;1;3;3;1];
 Z = [1;1;1;1;1];
 hold on;
 plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy e with z = 0
 plot3(X,Y,Z+2,'k','Linewidth',3); % draw a square in the xy plane with z = 1
 set(gca,'View',[32,25]); % set the azimuth and elevation of the plot [-28 35]
  for k=1:length(X)-1
     plot3([X(k);X(k)],[Y(k);Y(k)],[1;3],'k','Linewidth',3);
  end
plot3(pos(:,1),pos(:,2),pos(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5); hold on;axis off
view([92 10])
plot3(firposgrid(:,1),firposgrid(:,2),firposgrid(:,3),'.r', 'markersize', 25);
axis off
title('Plane field 3')
% Border field
load('trj_2')
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction;
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch);
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az);
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch);
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% PI_Az layer
basefreq=1;
speed_az=s_az';
dendosc=[];
basephase=0;
dendphase=0;
dt=0.01;
betaa=2;
dendfreq=[];
dendfreq=(basefreq*(ones(length(pos)-1,n_Az)))+betaa*repmat(speed_az,1,n_Az).*cos(repmat(theta_pref_Az,length(pos)-1,1)-repmat(headdir_az(1:end-1),1,n_Az));
piosc_az=[];
X = zeros(n_Az,1); Y = ones(n_Az,1);
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    basephase=basephase+2*pi*basefreq*dt;
    baseosc=sin(basephase);
    baseosc_arr(ii) = baseosc;
    dendosc=sin(dendphase);
    baseosc=sin(basephase);
    piosc_az(:,ii)=dendosc';
end
% PI_pitch layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
RBP = 15;
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1); 
betaa2=RBP*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=sin(dendphase);
    piosc_pitch(:,ii)=dendosc';
end
% Stacking piosc_Az and piosc_pitch
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=(abs(piosc_tot)>0.9);
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
osccupancy_time_bin_flag = nan(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
    osccupancy_time_bin_flag(q1,q2,q3) = 1;
end
osccupancy_time_bin_flag = fliplr(osccupancy_time_bin_flag);
% Border 1
load('lahn_wt_11')
neuron_ind = 40;
w=(T(neuron_ind,:));w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.7;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.1*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.3)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([26 43])
title('Border 1')
% Border 2
load('lahn_wt_12')
neuron_ind = 13;
w=(T(neuron_ind,:));w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=(ot);
thresh=max(ot)*.7;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.15*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.3)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([70 35])
title('Border 2')
% Border 3
load('lahn_wt_11')
neuron_ind = 42;
w=(T(neuron_ind,:));w = w'; 
ot=w'*piosc_thresh_tot; ot=ot';
ot=abs(ot);
thresh=max(ot)*.7;
firr=[];
firr=find((ot)>thresh);
firposgrid=pos(firr,:);
[fx,fy,fz] = meshgrid(1:1/res:3);
firingmap = zeros(length(fx),length(fx),length(fx));
gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
roundinggridpoint = round(gridpoint);
firposround = round(firposgrid);
firingvalue = (ot(firr));
for ii = 1:length(firposgrid)
    [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
    [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
    [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
    firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
end
firingmap2 = firingmap./(osccupancy_time_bin+eps);
firingmap2=fliplr(firingmap2);
sigma=3;
b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
thresh=0.1*max(max(max((b))));
b(abs(b)<thresh)=nan;
b2=imrotate(b/max(max(max(b))),90);
figure
h = slice(b2, [], [], 1:size(b,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.3)
colormap(jet)
axis off
view([-56 19])
hold on
X = [1;size(b,1);size(b,1);1;1];
Y = [1;1;size(b,2);size(b,2);1];
Z = [1;1;1;1;1];
minpos=1; maxpos=3;
pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
end
view([29 55])
title('Border 3')
train_flag = 0;
if train_flag
    PI1d = piosc_thresh_tot;
    [N K] = size(PI1d); %N --> Dimension    K---> # of samples
    % PI1d=(abs(PI1d)>0.9).*PI1d;
    PI1d=removemean(PI1d);
    alphaa = 0.0001/K;
    betaaa = 0.00001/K;
    output_neuron_nmbr = 40;
    maxiter = 2000000;
    [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaaa, output_neuron_nmbr, maxiter);
    bs_arr = [];
    for ind = 1:output_neuron_nmbr
        neuron_ind = ind;
        w=(T(neuron_ind,:));w = w'; 
        ot=w'*piosc_thresh_tot; ot=ot';
        ot=(ot);
        thresh=max(ot)*.7;
        firr=[];
        firr=find((ot)>thresh);
        firposgrid=pos(firr,:);
        [fx,fy,fz] = meshgrid(1:1/res:3);
        firingmap = zeros(length(fx),length(fx),length(fx));
        gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
        roundinggridpoint = round(gridpoint);
        firposround = round(firposgrid);
        firingvalue = (ot(firr));
        for ii = 1:length(firposgrid)
            [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
            [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
            [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
            firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
        end
        firingmap2 = firingmap./(osccupancy_time_bin+eps);
        firingmap2=fliplr(firingmap2);
        bs_arr(ind) = bs_compute(firingmap2);
    end    
    q = find(bs_arr>0.52);   
    figure
    for q_ind = 1:length(q)
        neuron_ind = q(q_ind);
        w=(T(neuron_ind,:));w = w'; 
        ot=w'*piosc_thresh_tot; ot=ot';
        ot=abs(ot);
        thresh=max(ot)*.7;
        firr=[];
        firr=find((ot)>thresh);
        firposgrid=pos(firr,:);        
        [fx,fy,fz] = meshgrid(1:1/res:3);
        firingmap = zeros(length(fx),length(fx),length(fx));
        gridpoint = [reshape(fx,prod(size(fx)),1) reshape(fy,prod(size(fx)),1) reshape(fz,prod(size(fx)),1)];
        roundinggridpoint = round(gridpoint);
        firposround = round(firposgrid);
        firingvalue = (ot(firr));
        for ii = 1:length(firposgrid)
            [~,q1]=min(abs(firposgrid(ii,1)-fx(1,:)));
            [~,q2]=min(abs(firposgrid(ii,2)-fx(1,:)));
            [~,q3]=min(abs(firposgrid(ii,3)-fx(1,:)));
            firingmap(q1,q2,q3) = firingmap(q1,q2,q3) + 1;
        end
        firingmap2 = firingmap./(osccupancy_time_bin+eps);
        firingmap2=fliplr(firingmap2);        
        sigma=3;
        b = imgaussfilt3(firingmap2, sigma); spikes_smooth=b;
        thresh=0.1*max(max(max((b))));
        b(abs(b)<thresh)=nan;
        b2=imrotate(b/max(max(max(b))),90);
        subplot(5,8,q_ind)
        h = slice(b2, [], [], 1:size(b,3));
        set(h, 'EdgeColor','none', 'FaceColor','interp')
        alpha(.3)
        colormap(jet)
        axis off
%         axis equal
        view([-56 19])
        hold on
        X = [1;size(b,1);size(b,1);1;1];
        Y = [1;1;size(b,2);size(b,2);1];
        Z = [1;1;1;1;1];
        minpos=1; maxpos=3;
        pos_norm = (pos-minpos)/(maxpos-minpos); pos_rescale= (max(X)-min(X))*pos_norm +min(X);
        plot3(pos_rescale(:,1),pos_rescale(:,2),pos_rescale(:,3),'Color',0.8*ones(1,3),'Linewidth',0.5);
        plot3(X,Y,Z,'k','Linewidth',3);   % draw a square in the xy plane with z = 0
        plot3(X,Y,Z+size(b,3)-1,'k','Linewidth',3); % draw a square in the xy plane with z = 1
        for k=1:length(X)-1
            plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',3);
        end
        view([26 43])
    end
end