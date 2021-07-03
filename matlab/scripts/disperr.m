
function [] = disperr(u, v)

disp(strcat((['Error |' inputname(1) '-' inputname(2) '|/|' inputname(1) '| = ' num2str(norm(u-v)/norm(u))])));

end
