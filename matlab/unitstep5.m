n = -10:10;
u = (n >= 0);
u_shift = (n >= 5);

y = u - u_shift;

stem(n, y, 'filled')
title('Pulso rectangular: u[n] - u[n-5]')
grid on