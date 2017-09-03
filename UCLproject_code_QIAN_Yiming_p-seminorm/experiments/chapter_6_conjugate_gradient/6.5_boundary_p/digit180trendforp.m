rand('seed',0);
%%Get the group 1 and 2(each has 90 points)
load distance_binaryalphadigs_data;
load binaryalphadigs_labels
labels = class_labels;

label1 = find(labels=='1');
label11 = find(labels=='1');
label111 = find(labels=='7');

label2 = find(labels=='6');
label22 = find(labels=='6');
label222 = find(labels=='9');

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:30);
label2 = label2(1:30);

label11 = label1(1:30);
label22 = label2(1:30);

label111 = label1(1:30);
label222 = label2(1:30);

%% the total labeled points is 16 means that label1 and label2 are 8 for each
%% the first 8 points is labeled (fl) and the others are unlabeled (fu)
totallabel = [label1(1:3);label11(1:3);label111(1:2);label2(1:3);label22(1:3);label222(1:2);label1(4:30);label11(4:30);label111(3:30);label2(4:30);label22(4:30);label222(3:30)];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K=6) 

%% find the nearest two points(not connect by self)
K=6;
usedistance1 = usedistance;
for i=1:180
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(180,K);

for i=1:180
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(180,180);
for i=1:180
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% suppose label1 is 1 and 0 for label2


fl=zeros(16,1);
fl(1:8)=1;
fl(9:16)=0;

SIGrange= [1e-5];
prange = [ 1.1, 1.2, 1.4, 1.6, 1.8, 2, 4, 8, 12, 16, 24, 32, 64, 128];
Mat = zeros(length(prange)*length(SIGrange),6);
kk=0;

for iter=1:length(prange)

for SIG=SIGrange    
    
kk=kk+1;    
    
rand('seed',0);    
    
    
fprintf('1\n');
%% using conjugate gradient to do the compute
f = @diffforCGpnew;
X = ones(164,1);
%check = 0;
Xg = zeros(164,1);
gap=1;
times = 2*length(X);
p=prange(iter);

tic;
[X, fX, i] = sigminimizep(X, f,fl,A,p,SIG, 10000);
timeforiter = toc;
iterations = i;
    
XA = repmat(X,1,length(X));
XB = abs(XA-XA');
AB = XB.*A(17:180,17:180);
energy = (1/2)*sum(sum(AB.^p));


fu = X;


fprintf('2\n');


fu = fu>0.5;

%% the true labels for unlabeled points
futrue = zeros(164,1);
futrue(1:82)=1;
futrue(83:164)=0;

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

figure,
plot(log2(Result(:,1)'),Result(:,5)');
hold on;
title('The iteration for p value in (1, inf) of convergence');
xlabel('log(base2) value of p');
ylabel('the iteration');
hold off;

figure,
plot(log2(Result(:,1)'),Result(:,3)','r--');
hold on;
plot(log2(Result(:,1)'),Result(:,3)','g*');
hold on;
title('The accuracy for p value in (1, inf) of convergence');
xlabel('log(base2) value of p');
ylabel('the accuracy');
hold off;


