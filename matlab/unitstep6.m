clear; clc; close all

t = -2:0.001:2;

% Aproximación de delta(t): pulso rectangular de área 1
Delta = 0.001;                 % ancho del pulso, probar con 0.5, 0.05, 0.01
delta_aprox = (t>=0 & t<=Delta) / Delta;

% Integral acumulada de delta(t), aproxima u(t)
u_aprox = cumtrapz(t, delta_aprox);

figure

subplot(3,1,1)
plot(t, delta_aprox, 'LineWidth', 2)
grid on
title('\delta(t) aproximado usando un pulso rectangular')
xlabel('t')
ylabel('\delta_\Delta(t)')
ylim([0 max(delta_aprox)*1.2])

subplot(3,1,2)
plot(t, u_aprox, 'LineWidth', 2)
grid on
title('Integral acumulada de \delta(t): aparece el unit step continuo u(t)')
xlabel('t')
ylabel('u(t)')
ylim([-0.1 1.2])

% Propiedad de muestreo: x(t)delta(t) = x(0)delta(t)
x = 1 + 0.4*sin(2*pi*t) + 0.2*cos(5*pi*t);
producto = x .* delta_aprox;

subplot(3,1,3)
plot(t, x, '--', 'LineWidth', 1.5)
hold on
plot(t, producto, 'LineWidth', 2)
grid on
title('Producto x(t)\delta(t): solo importa x(0)')
xlabel('t')
legend('x(t)', 'x(t)\delta_\Delta(t)')