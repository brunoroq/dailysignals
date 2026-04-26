%Sumatoria de impulsos: x[n] = cumsum(x[k]delta[n-k]) 
n = -5:5;
x = [0 0 1 2 3 2 1 0 0 0 1]; % Señal cualquiera
stem(n,x,'filled')
title('señal x[n]')
grid on
%Resultado, señal original. 