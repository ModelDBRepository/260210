function gscore = gridscore(c)
c60=(imrotate(c,60));
a = 0.5*(size(c60,1) - size(c,1));
b = size(c60,1) - a;
c60_2 = c60(a+1:b,a+1:b);
r60 = corrcoef(c,c60_2);

c120=(imrotate(c,120));
c120_2 = c120(a+1:b,a+1:b);
r120 = corrcoef(c,c120_2);

c30=(imrotate(c,30));
c30_2 = c30(a+1:b,a+1:b);
r30 = corrcoef(c,c30_2);

c90=(imrotate(c,90));
r90 = corrcoef(c,c90);

c150=(imrotate(c,150));
c150_2 = c150(a+1:b,a+1:b);
r150 = corrcoef(c,c150_2);

gscore = min([r60(1,2) r120(1,2)]) - max([r30(1,2) r90(1,2) r150(1,2)]);
end