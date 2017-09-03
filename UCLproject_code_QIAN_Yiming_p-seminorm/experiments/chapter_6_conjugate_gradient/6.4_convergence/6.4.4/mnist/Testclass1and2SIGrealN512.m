rand('seed',0);
%%Get the data 1 and 2(each has 256 points)
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==1);
label2 = find(labels==2);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:256);
label2 = label2(1:256);

%% the total labeled points is 16 means that label1 and label2 are 8 for each
%% the first 8 points is labeled (fl) and the others are unlabeled (fu)
totallabel = [label1(1:8);label2(1:8);label1(9:256);label2(9:256)];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K=6) 

%% find the nearest two points(not connect by self)
K=6;
usedistance1 = usedistance;
for i=1:512
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(512,K);

for i=1:512
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(512,512);
for i=1:512
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% suppose label1 is 1 and 0 for label2


fl=zeros(16,1);
fl(1:8)=1;
fl(9:16)=0;

SIGrange= [0.1, 1e-2, 1e-3, 1e-4, 1e-5];
prange = [ 1.6,1.8, 2, 4, 8 ];
Mat = zeros(length(prange)*length(SIGrange),6);
kk=0;

for iter=1:length(prange)

for SIG=SIGrange    
    
kk=kk+1;    
    
rand('seed',0);    
    
    
fprintf('1\n');
%% using conjugate gradient to do the compute
f = @diffforCGpnew;
X = ones(496,1);
%check = 0;
Xg = zeros(496,1);
gap=1;
times = 2*length(X);
p=prange(iter);

tic;
[X, fX, i] = sigminimizep(X, f,fl,A,p,SIG, 10000);
timeforiter = toc;
iterations = i;
    
XA = repmat(X,1,length(X));
XB = abs(XA-XA');
AB = XB.*A(17:512,17:512);
energy = (1/2)*sum(sum(AB.^p));


fu = X;


fprintf('2\n');


fu = fu>0.5;

%% the true labels for unlabeled points
futrue = zeros(496,1);
futrue(1:248)=1;
futrue(249:496)=0;

%% compare the accuracy
final = (futrue==fu);
accuracy = sum(final)/length(final);
fprintf('The accuracy is %.6f   The energy is %.6f  The iteration is %.6f  The time is %.6f\n', accuracy,energy,iterations,timeforiter);
%%fprintf('The accuracy is %d\n', sum(final));
%%fprintf('The accuracy is %.2f\n', length(final));
Mat(kk,1)=p;
Mat(kk,2)=SIG;
Mat(kk,3)=accuracy;
Mat(kk,4)=energy;
Mat(kk,5)=iterations;
Mat(kk,6)=timeforiter;
end

end

Result = Mat;
for j=1:size(Result,1)
    fprintf('The p is %.6f  The SIG is %.6f  The accuracy is %.6f   The energy is %.6f  The iteration is %.6f  The time is %.6f\n', Result(j,1),Result(j,2),Result(j,3),Result(j,4),Result(j,5),Result(j,6));


end

