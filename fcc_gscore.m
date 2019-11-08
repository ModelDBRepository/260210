function [mean_gscore gscore_arr] = fcc_gscore(c)
tic
%% Slice autocorrelation map and find reference plane
close all
theta_az_arr = -90:15:90;
theta_p_arr = theta_az_arr;
gscore_mat = [];
pt = median(1:size(c,1))*ones(1,3);
slice_len = 20;
for az_loop = 1:length(theta_az_arr)
    for p_loop = 1:length(theta_p_arr)
        fprintf('\n%d/%d,%d/%d',az_loop,length(theta_az_arr),p_loop,length(theta_p_arr))
        theta_az = theta_az_arr(az_loop);
        theta_p = theta_p_arr(p_loop);
        vec = [cosd(theta_az)*sind(theta_p) sind(theta_az)*sind(theta_p) cosd(theta_p)]';
        [slice2, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec(1),vec(2),vec(3),slice_len);
        slice2(isnan(slice2))=0;
        figure; imagesc(slice2); colormap(jet)
        gscore = gridscore(slice2);
        gscore_mat(az_loop,p_loop) = gscore;
    end
end
[az_max_ind p_max_ind val]=max2(gscore_mat);
az_max = theta_az_arr(az_max_ind(1))
p_max = theta_p_arr(p_max_ind(1));
vec_ref = [cosd(az_max)*sind(p_max) sind(az_max)*sind(p_max) cosd(p_max)]';
%% First plane
theta_az_vec_arr = -90:90;
theta_p_vec_arr = theta_az_vec_arr;
ang_mat_vec1ref = [];
for ii=1:length(theta_az_vec_arr)
    for jj=1:length(theta_p_vec_arr)
        fprintf('\n%d/%d,%d/%d',ii,length(theta_az_vec_arr),jj,length(theta_p_vec_arr))
        vec1 = [cosd(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) sind(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) cosd(theta_p_vec_arr(jj))]';        
        ang_mat_vec1ref(ii,jj) = rad2deg(acos(dot(vec_ref,vec1)));
    end
end
alphaa = 72;
mat_sim_72_vec1ref = abs(ang_mat_vec1ref-alphaa);
% figure; surf(mat_com_vec123);
[q1 q2] = find(mat_sim_72_vec1ref==(min(min(mat_sim_72_vec1ref))));
% Compute gridness for all planes that are 72 deg from ref plane
gscore_mat_ref1 = [];
for ref1_arr = 1:length(q1)
    fprintf('\n%d/%d',ref1_arr,length(q1))
    ind = ref1_arr;
    thata_az_sel = theta_az_vec_arr(q1(ind));
    theta_p_sel = theta_p_vec_arr(q2(ind));
    vec1 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';
    % rad2deg(acos(dot(vec_ref,vec1)))
    [slice2, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec1(1),vec1(2),vec1(3),slice_len);
    slice2(isnan(slice2))=0;
    %         figure; imagesc(slice2); colormap(jet)
    gscore = gridscore(slice2);
    gscore_mat_ref1(ref1_arr) = gscore;
end
[~,ref1_ind] = max(gscore_mat_ref1);
thata_az_sel = theta_az_vec_arr(q1(ref1_ind));
theta_p_sel = theta_p_vec_arr(q2(ref1_ind));
vec1 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';
%% Second plane
for ii=1:length(theta_az_vec_arr)
    for jj=1:length(theta_p_vec_arr)
        fprintf('\n%d/%d,%d/%d',ii,length(theta_az_vec_arr),jj,length(theta_p_vec_arr))
        vec2 = [cosd(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) sind(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) cosd(theta_p_vec_arr(jj))]';        
        ang_mat_vec12(ii,jj) = rad2deg(acos(dot(vec1,vec2)));      
        ang_mat_vecref2(ii,jj) = rad2deg(acos(dot(vec_ref,vec2)));
    end
end
alphaa = 72;
mat_sim_72_vec12 = abs(ang_mat_vec12-alphaa);
mat_sim_72_vecref2 = abs(ang_mat_vecref2-alphaa);
mat_com_vec12ref = abs(mat_sim_72_vec12+mat_sim_72_vecref2);
% figure; surf(mat_com_vec123);
[q1 q2] = find(mat_com_vec12ref==(min(min(mat_com_vec12ref))));

gscore_mat_ref2 = [];
for ref2_arr = 1:length(q1)
    fprintf('\n%d/%d',ref2_arr,length(q1))
    ind = ref2_arr;
    thata_az_sel = theta_az_vec_arr(q1(ind));
    theta_p_sel = theta_p_vec_arr(q2(ind));
    vec2 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';
    % rad2deg(acos(dot(vec_ref,vec1)))
    [slice2, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec2(1),vec2(2),vec2(3),slice_len);
    slice2(isnan(slice2))=0;
    %         figure; imagesc(slice2); colormap(jet)
    gscore = gridscore(slice2);
    gscore_mat_ref2(ref2_arr) = gscore;
end
[~,ref2_ind] = max(gscore_mat_ref2);
thata_az_sel = theta_az_vec_arr(q1(ref2_ind));
theta_p_sel = theta_p_vec_arr(q2(ref2_ind));
vec2 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';
%% Third plane
[vec3] = ref_vec3_fit(vec1,vec2,vec_ref);
% ang_mat_vec13 = []; ang_mat_vec23 = []; ang_mat_vecref3 = [];
% for ii=1:length(theta_az_vec_arr)
%     for jj=1:length(theta_p_vec_arr)
%         fprintf('\n%d/%d,%d/%d',ii,length(theta_az_vec_arr),jj,length(theta_p_vec_arr))
%         vec3 = [cosd(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) sind(theta_az_vec_arr(ii))*sind(theta_p_vec_arr(jj)) cosd(theta_p_vec_arr(jj))]';        
%         ang_mat_vec13(ii,jj) = rad2deg(acos(dot(vec1,vec3)));
%         ang_mat_vec23(ii,jj) = rad2deg(acos(dot(vec2,vec3)));
%         ang_mat_vecref3(ii,jj) = rad2deg(acos(dot(vec_ref,vec3)));
%     end
% end
% alphaa = 72;
% mat_sim_72_vec13 = abs(ang_mat_vec13-alphaa);
% mat_sim_72_vec23 = abs(ang_mat_vec23-alphaa);
% mat_sim_72_vecref3 = abs(ang_mat_vecref3-alphaa);
% mat_com_vec123ref = abs(0*mat_sim_72_vec13+mat_sim_72_vec23+mat_sim_72_vecref3);
% % figure; surf(mat_com_vec123);
% [q1 q2] = find(mat_com_vec123ref==(min(min(mat_com_vec123ref))));
% gscore_mat_ref3 = [];
% for ref3_arr = 1:length(q1)
%     fprintf('\n%d/%d',ref3_arr,length(q1))
%     ind = ref3_arr;
%     thata_az_sel = theta_az_vec_arr(q1(ind));
%     theta_p_sel = theta_p_vec_arr(q2(ind));
%     vec3 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';
%     % rad2deg(acos(dot(vec_ref,vec1)))
%     [slice2, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec3(1),vec3(2),vec3(3),slice_len);
%     slice2(isnan(slice2))=0;
%     %         figure; imagesc(slice2); colormap(jet)
%     gscore = gridscore(slice2);
%     gscore_mat_ref3(ref3_arr) = gscore;
% end
% % [val,ref3_ind] = max(gscore_mat_ref3);
% [val,ref3_ind] = sort(gscore_mat_ref3,'descend');
% thata_az_sel = theta_az_vec_arr(q1(ref3_ind(1)));
% theta_p_sel = theta_p_vec_arr(q2(ref3_ind(1)));
% vec3 = [cosd(thata_az_sel)*sind(theta_p_sel) sind(thata_az_sel)*sind(theta_p_sel) cosd(theta_p_sel)]';

%% Compute HGS for each selected plane
[slice1, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec1(1),vec1(2),vec1(3),slice_len);
slice1(isnan(slice1))=0;
gscore1 = gridscore(slice1);

[slice2, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec2(1),vec2(2),vec2(3),slice_len);
slice2(isnan(slice2))=0;
gscore2 = gridscore(slice2);

[slice3, sliceInd,subX,subY,subZ] = extractSlice(c,pt(1),pt(2),pt(3),vec3(1),vec3(2),vec3(3),slice_len);
slice3(isnan(slice3))=0;
gscore3 = gridscore(slice3);
gscore_arr = [gscore1 gscore2 gscore3];   
mean_gscore = mean(gscore_arr);
toc
end