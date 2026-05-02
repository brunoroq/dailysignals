% Parámetros del circuito
R = 1e3;          % Resistencia (Ohm)
C = 1e-6;         % Capacitancia (Farad)
tau = R*C;        % Constante de tiempo: tau = RC

% Vector de tiempo
t = linspace(0, 5*tau, 1000);
% Estamos observando el sistema en un intervalo de varias constantes de tiempo

% Entrada: escalón unitario
vs = ones(size(t));
% Matemáticamente: v_s(t) = 1 para t >= 0

% Solución analítica de la ecuación diferencial:
% dv_c/dt + (1/RC)*v_c = (1/RC)*v_s
%
% Para entrada escalón, la solución es:
% v_c(t) = 1 - e^(-t/RC)

vc = 1 - exp(-t/tau);

% Esto viene de resolver:
% dv_c/dt = (1/RC)(1 - v_c)
%
% Es decir, la tasa de cambio depende de qué tan lejos está vc de 1

% Corriente usando ley de Ohm:
i = (vs - vc)/R;

% Esto corresponde a:
% i(t) = (v_s(t) - v_c(t)) / R
%
% Y también cumple:
% i(t) = C * dv_c/dt
%
% (puedes verificar derivando vc)

% ==============================
% Visualización
% ==============================

figure

subplot(3,1,1)
plot(t, vs)
title('Entrada v_s(t): escalón')
ylabel('Voltaje (V)')
grid on
% Representa la excitación del sistema

subplot(3,1,2)
plot(t, vc)
title('Salida v_c(t): solución de la ecuación diferencial')
ylabel('Voltaje (V)')
grid on
% Esta es la solución de:
% dv_c/dt + (1/RC)v_c = (1/RC)
%
% Observa:
% - crecimiento exponencial
% - límite en 1 (estado estacionario)

subplot(3,1,3)
plot(t, i)
title('Corriente i(t)')
xlabel('Tiempo (s)')
ylabel('Corriente (A)')
grid on
% Esta es:
% i(t) = (1/R)e^(-t/RC)
%
% Interpretación:
% - al inicio hay máxima corriente (capacitor descargado)
% - luego decae exponencialmente