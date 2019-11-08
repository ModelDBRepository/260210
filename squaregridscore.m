function gscore = squaregridscore(c)

c90=(imrotate(c,90));
r90 = corrcoef(c,c90);

c45=(imrotate(c,45));
a = 0.5*(size(c45,1) - size(c,1));
b = size(c45,1) - a;
c45_2 = c45(a+1:b,a+1:b);
r45 = corrcoef(c,c45_2);

c135=(imrotate(c,135));
c135_2 = c135(a+1:b,a+1:b);
r135 = corrcoef(c,c135_2);

gscore = r90(1,2) - max([r45(1,2) r135(1,2)]);
end