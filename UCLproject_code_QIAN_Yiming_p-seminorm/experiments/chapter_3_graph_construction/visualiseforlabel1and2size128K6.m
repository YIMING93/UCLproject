%% Get the data 1 and 2(each has 64 points)
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==1);
label2 = find(labels==2);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:64);
label2 = label2(1:64);

totallabel = [label1;label2];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K=6) 

%% find the nearest two points(not connect by self)
K=6;
usedistance1 = usedistance;
for i=1:128
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(512,K);

for i=1:128
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(128,128);
for i=1:128
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% visualise the KNN graph 
%%red is label1
%%blue is label2

s = sum(A);
di = diag(sparse(1./s));
m = di*A;
[vector,eigenval] = eigs(m,6);
gplot(A,vector(:,2:3),'k');
hold on
gplot(A,vector(:,2:3),'bo');
gplot(A(1:64,1:64),vector(:,2:3),'ro');