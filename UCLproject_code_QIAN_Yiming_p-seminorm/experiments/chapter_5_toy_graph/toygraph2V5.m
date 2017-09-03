%% The closed form solution for toy graph model2 node 5 p-voltages

prange = [1.1, 1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
results = zeros(1,17);
i=0;

for p=prange

i=i+1;
% the mathsmatic algorithm solution for toy graph 3 
f = (2-3^(1/(p-1)))/(2+3^(1/(p-1)));
results(i)=f;

end

%plot(prange, results, 'k--');
%hold on;

%% conjugate gradient values of MAD in toy graph model2
A=[ 0,0,0,0,1, 0,0,0;
    0,0,0,0,0, 1,0,0;
    0,0,0,0,0, 0,1,0;
    0,0,0,0,0, 0,0,1;
    1,0,0,0,0, 1,1,1;
    
    0,1,0,0,1, 0,0,0;
    0,0,1,0,1, 0,0,0;
    0,0,0,1,1, 0,0,0;
   ];

results2 = zeros(1,17);
j=0;
for p=prange
j=j+1;
f = @diffforCGpnew;
X = ones(4,1);
flc=[1,-1,-1,-1]';
[X, fX, i] = minimizep(X, f,flc,A,p, 10000);
CGX = X;
results2(j)=CGX(1);

end

plot(prange, results2, 'r*',prange, results, 'b--');
hold on;
title('The voltage value of node 5 in minimum P(v) condition');
xlabel('p value');
ylabel('the node 5 voltage');
legend('CG','Theory');


