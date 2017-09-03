rand('seed',0);
%%Get the data 1 and 2(each has 128 points) The example structure of
%%conjugate gradient of p=2
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==1);
label2 = find(labels==2);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:128);
label2 = label2(1:128);

%% the total labeled points is 16 means that label1 and label2 are 8 for each
%% the first 8 points is labeled (fl) and the others are unlabeled (fu)
totallabel = [label1(1:4);label2(1:4);label1(5:128);label2(5:128)];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K=6) 

%% find the nearest two points(not connect by self)
K=6;
usedistance1 = usedistance;
for i=1:256
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(256,K);

for i=1:256
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(256,256);
for i=1:256
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% suppose label1 is 1 and 0 for label2


fl=zeros(8,1);
fl(1:4)=1;
fl(5:8)=0;

fprintf('1\n');
%% using conjugate gradient to do the compute
f = @diffforCG;
X = ones(248,1);
%check = 0;
Xg = zeros(248,1);
gap=1;
times = 2*length(X);

while (times>0)
    Xg = X;
    [X, fX, i] = minimize(X, f,fl,A, 100);
    gap = sum((X-Xg).^2);
    if (gap<1e-20)
        times = -1;
    end
    times = times-1;

end

fu = X;


fprintf('2\n');


fu = fu>0.5;

%% the true labels for unlabeled points
futrue = zeros(248,1);
futrue(1:124)=1;
futrue(125:248)=0;

%% compare the accuracy
final = (futrue==fu);
accuracy = sum(final)/length(final);
fprintf('The accuracy is %.3f\n', accuracy);
%%fprintf('The accuracy is %d\n', sum(final));
%%fprintf('The accuracy is %.2f\n', length(final));