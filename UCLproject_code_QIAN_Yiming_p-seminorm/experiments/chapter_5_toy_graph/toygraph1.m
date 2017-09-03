
%% using harmonic function to do the test of conjugate gradient values of MAD in toy graph model1
A=[ 0,0,1,0;
    0,0,0,1;
    1,0,0,1;
    0,1,1,0 ];


p=8;
f = @diffforCGpnew;
X = ones(2,1);
flc=[1,-1]';
[X, fX, i] = minimizep(X, f,flc,A,p, 10000);
CGX = X


flh=[ 1,0;
      0,1 ];
value= [1,-1]';
  
[fu, fu_CMN] = harmonic_function(A, flh);
HMX = fu*value





MAD = sum(abs(CGX-HMX))/length(CGX);

MAD=MAD



