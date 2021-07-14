
function [] = disperr2d(u, v)

disp(strcat((['immse (' inputname(1) ',' inputname(2) ') = ' num2str(immse(u,v))])));

end
