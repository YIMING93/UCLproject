function [f, df, ddf] = diffforCGp(x,l,A,p)

% improved diffforCG function for F(x) first order and second order differential
% value for p>2 condition

total = [l;x];


D = length(x);
x1=repmat(total,1,length(total));
x2=x1';
X=(x1-x2).^(p);
DX = (x1-x2).^(p-1);
DDX = (x1-x2).^(p-2);

f = (1/p)*sum(sum(X.*A));

if nargout > 1
  df = zeros(D, 1);
  sdf = 2*sum((DX.*A),2);
  df = sdf((length(l)+1):length(total));
  
end

if nargout > 2
  ddt = (-2)*(p-1)*DDX.*A;
  ddf = ddt((length(l)+1):length(total),(length(l)+1):length(total));
end 