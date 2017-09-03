function [f, df, ddf] = diffforCGpnew(x,l,A,p)

% improved diffforCGp function for F(x) first order and second order differential
% value for p in (1, inf)

total = [l;x];


D = length(x);
x1=repmat(total,1,length(total));
x2=x1';

%%
%%x1 k
%%x2 m
ORG = (x1-x2);
ABS = abs(x1-x2);
SIGNc = ORG-ABS;
SIGN = (SIGNc==0);
SIGN = 2*(SIGN-0.5);


X=(ABS).^(p);
DX = (ABS).^(p-1);
DDX = (ABS).^(p-2);

f = (1/2)*sum(sum(X.*A));

if nargout > 1
  df = zeros(D, 1);
  sdf = p*sum((DX.*A.*SIGN),2);
  df = sdf((length(l)+1):length(total));
  
end

if nargout > 2
  ddt = (-1)*(p)*(p-1)*DDX.*A;
  ddf = ddt((length(l)+1):length(total),(length(l)+1):length(total));
end 

