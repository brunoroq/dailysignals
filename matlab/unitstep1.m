n=0:1:20;
x=10*(n>=0)-5*(n>=5)-10*(n>=10)+5*(n>=15);
stem(n,x);
ylabel('Time Sample')
xlabel('Amplitude')
axis([0 20 -10 20])