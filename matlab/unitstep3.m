n = -10:10;
delta = (n == 0);

u_calc = cumsum(delta); % unit step como suma acumulada en tiempo discreto

stem(n, u_calc,'filled')
title('u[n] como suma de \delta[n]')
grid on