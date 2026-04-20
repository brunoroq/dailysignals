%Multiplicar una señal por e^-jωt “la gira” y si eliges la frecuencia correcta, se vuelve constante.
t = linspace(0,1,1000);

% señal: coseno de 5 Hz
x = cos(2*pi*5*t);

% probamos distintas frecuencias
w_test = [3, 5, 7];

figure;
for k = 1:length(w_test)
    w = w_test(k);
    y = x .* exp(-1j*2*pi*w*t);
    
    subplot(3,1,k)
    plot(real(y))
    title(['Frecuencia de prueba: ', num2str(w), ' Hz'])
end