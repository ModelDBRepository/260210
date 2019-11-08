% This code shows the details of hierarchical layers in the neural network model of 3D spatial representations.
% Detailed implementation with many examples are given in another code named "main_code".
% Model is driven by flight trajectory. Trajectory should be given as a
% matrix of form Nx3, where N is the total number of time samples in the
% trajectory which has 3 dimensions such as X,Y and Z respectively embedded
% in each column. 
clc
clear all
close all
% Load the trajectory here
%An example trajectory is loaded below
load('3d_trj_cuboid_1_pitch_stati')
% Head direction (hd) coding layer of the model
% Two parallel layers are considered, one for azimuth and another for pitch
n=100; % #total number of hd cells
n_Az_fraction = 70/100; n_pitch_fraction = 1-n_Az_fraction; % numner of az and pitch neurons are separated in 7:3 ratio
n_Az = n_Az_fraction*n;
n_pitch = floor(n_pitch_fraction*n)-1;
pos_az = pos(:,1:2);
delx = diff(pos(:,1)); delx(end+1)=0;
dely = diff(pos(:,2)); dely(end+1)=0;
delz = diff(pos(:,3)); delz(end+1)=0;
theta_az = rad2deg(atan2(dely,delx));headdir_az=deg2rad(theta_az);%Compute azimuth angle
theta_pitch = rad2deg(atan2(delz,sqrt(delx.^2+dely.^2)));headdir_pitch=deg2rad(theta_pitch); %Compute pitch angle
dth_Az=360/n_Az;
dth_pitch=360/n_pitch;
theta_pref_deg_Az=0:dth_Az:360-dth_Az; theta_pref_Az=deg2rad(theta_pref_deg_Az); %azimuth preferred direction of neurons
theta_pref_deg_pitch=0:dth_pitch:360-dth_pitch;  theta_pref_pitch=deg2rad(theta_pref_deg_pitch); %pitch preferred direction of neurons
% Speed estimation
s=speedpos(pos);
s_az=s;s_pitch=s;
% Azimuth oscillatory path integration layer
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
% Start loop for path integration.
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
% Pitch oscillatory path integration layer
speed_pitch=s_pitch';
dendosc=[];
basephase=0;
dendphase=0;
dendfreq=[];
Xp = zeros(n_pitch,1); Yp = ones(n_pitch,1); %Xarr=[]; Yarr=[];
betaa2=15*betaa/100;
dendfreq=(basefreq*(ones(length(pos)-1,n_pitch)))+betaa2*repmat(speed_pitch,1,n_pitch).*cos(repmat(theta_pref_pitch,length(pos)-1,1)-repmat(headdir_pitch(1:end-1),1,n_pitch));
piosc_pitch=[];
dendphase_pitch = [];
% Start loop for path integration.
for ii=1:length(pos)-1
    dendphase=dendphase+2*pi*dendfreq(ii,:)*dt;
    dendosc=cos(dendphase);
    piosc_pitch(:,ii)=dendosc';
    dendphase_pitch(:,ii)=dendphase';
end
% Concatentating azimuth and pitch path integration values
piosc_tot = [piosc_az ;piosc_pitch];
piosc_thresh_tot=((piosc_tot)>0.9).*piosc_tot; % Thresholding the path integration values
% Occupancy of the trajectory
res = 20;
[fx,fy,fz] = meshgrid(1:1/res:3);
osccupancy_time_bin = zeros(size(fx));
[pgridx, pgridy,pgridz] = meshgrid(linspace(1,3,size(fx,1)));
for ii = 1:size(pos,1)
    ii
    [~,q1]=min(abs(pos(ii,1)-pgridx(1,:)));
    [~,q2]=min(abs(pos(ii,2)-pgridx(1,:)));
    [~,q3]=min(abs(pos(ii,3)-pgridx(1,:)));
    osccupancy_time_bin(q1,q2,q3) = osccupancy_time_bin(q1,q2,q3)+1;
end
% Train anti-hebbian network
PI1d = piosc_thresh_tot; %Input to the network
[N K] = size(PI1d); %N --> Dimension    K---> # of samples
PI1d=removemean(PI1d);
alphaa = 0.0001/K; %Afferent weights learning rate
betaa = 0.00001/K; %Lateral weights learning rate
output_neuron_nmbr = 40; %Total number of neurons in the network
maxiter = 2000000; %Max iteration for training
[T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);
% Compute the network output
figure
for neuron_index = 1:output_neuron_nmbr
    w = T(neuron_index,:)'; %Transformation weight of the network
    ot=w'*piosc_thresh_tot; ot=ot'; %Output of the network
    ot=abs(ot);
    thresh=max(ot)*.75;
    firr=[];
    firr=find((ot)>thresh);
    firposgrid=pos(firr,:);
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
    thresh=0.15*max(max(max((b))));
    b(abs(b)<thresh)=nan;
    b2=imrotate(b/max(max(max(b))),90);
    subplot(5,8,neuron_index)
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
        plot3([X(k);X(k)],[Y(k);Y(k)],[1;size(b,3)],'k','Linewidth',2);
    end
    view([-50 34])
end