function speed = speedpos(pos)
speed=[];
for ii=2:length(pos)
    speed=[speed pdist2(pos(ii-1,:),pos(ii,:))];
end
end