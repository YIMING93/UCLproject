function [f, df, ddf] = diffforCG(x,l,A)

% diffforCG function for F(x) first order and second order differential
% value for p=2 condition

total = [l;x];


D = length(x);
x1=repmat(total,1,length(total));
x2=x1';
X=(x1-x2).^2;
DX = x1-x2;

f = (1/2)*sum(sum(X.*A));

if nargout > 1
  df = zeros(D, 1);
  sdf = 2*sum((DX.*A),2);
  df = sdf((length(l)+1):length(total));
  
end

if nargout > 2
  ddf = zeros(D,D);
  ddf = -2*A((length(l)+1):length(total),(length(l)+1):length(total));
end 