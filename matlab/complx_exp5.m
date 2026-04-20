%Un sistema es un filtro de exponenciales
t = linspace(0,1,1000);

x1 = exp(1j*2*pi*2*t);
x2 = exp(1j*2*pi*20*t);

% filtro simple (promedio móvil)
b = ones(1,20)/20;

y1 = filter(b,1,x1);
y2 = filter(b,1,x2);

figure;
subplot(2,1,1)
plot(real(y1))
title('Frecuencia baja (pasa)')

subplot(2,1,2)
plot(real(y2))
title('Frecuencia alta (se atenúa)')