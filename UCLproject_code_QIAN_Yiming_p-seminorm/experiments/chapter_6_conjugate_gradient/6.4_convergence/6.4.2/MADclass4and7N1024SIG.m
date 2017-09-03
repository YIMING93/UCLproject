rand('seed',0);
%%Get the data 4 and 7(each has 512 points)
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==4);
label2 = find(labels==7);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:512);
label2 = label2(1:512);

%% the total labeled points is 16 means that label1 and label2 are 8 for each
%% the first 8 points is labeled (fl) and the others are unlabeled (fu)
totallabel = [label1(1:8);label2(1:8);label1(9:512);label2(9:512)];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K=6) 

%% find the nearest two points(not connect by self)
K=6;
usedistance1 = usedistance;
for i=1:1024
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(1024,K);

for i=1:1024
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(1024,1024);
for i=1:1024
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% suppose label1 is 1 and 0 for label2


fl=zeros(16,1);
fl(1:8)=1;
fl(9:16)=0;
fll=zeros(16,2);
fll(1:8,1)=1;
fll(9:16,2)=1;

SIGrange= [0.6,0.5,0.4,0.3,0.2, 0.1, 1e-2, 1e-3,1e-4,1e-5, 1e-6, 1e-7, 1e-8];
prange = [2];
Mat = zeros(length(prange)*length(SIGrange),5);
kk=0;

for iter=1:length(prange)

for SIG=SIGrange    
    
kk=kk+1;    
    
rand('seed',0);    
    
    
fprintf('1\n');
%% using harmonic function do the test of MAD
f = @diffforCGpnew;
X = ones(1008,1);
%check = 0;



p=prange(iter);

tic;
[X, fX, i] = sigminimizep(X, f,fl,A,p,SIG, 10000);
timeforiter = toc;
iterations = i;
    
XA = repmat(X,1,length(X));
XB = abs(XA-XA');
AB = XB.*A(17:1024,17:1024);
energy = (1/2)*sum(sum(AB.^p));

[fu, fu_CMN] = harmonic_function(A, fll);


MAD = sum(abs(fu(:,1)-X))/length(X);

fprintf('2\n');



Mat(kk,1)=p;
Mat(kk,2)=SIG;
Mat(kk,3)=MAD;
Mat(kk,4)=iterations;
Mat(kk,5)=timeforiter;
end

end

Result = Mat;
for j=1:size(Result,1)
    fprintf('The p is %.6f  The SIG is %.6f   The MAD is %.20f  The iteration is %.6f  The time is %.6f\n', Result(j,1),Result(j,2),Result(j,3),Result(j,4),Result(j,5));


end
