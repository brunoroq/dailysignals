t = linspace(0,2,2000);

x = exp(1j*2*pi*10*t) + exp(1j*2*pi*12*t);

figure;
subplot(2,1,1)
plot(real(x))
title('Señal en el tiempo')

subplot(2,1,2)
plot(abs(x))
title('Envolvente (beats)')