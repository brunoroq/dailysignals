n = -10:10;
delta = (n == 0);

u = (n >= 0);
u_shift = (n+2 >= 0);

delta_calc = u - u_shift;

stem(n, delta_calc, 'filled')
title('\delta[n] = u[n] - u[n+2]')
grid on