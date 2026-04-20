%Una señal es una suma de frecuencias.
t = linspace(0,1,1000);

x = cos(2*pi*3*t) + 0.5*cos(2*pi*7*t);

X = fft(x);

f = linspace(0,999,1000);

figure;
subplot(2,1,1)
plot(t,x)
title('Señal original')

subplot(2,1,2)
plot(f,abs(X))
xlim([0 20])
title('Espectro')