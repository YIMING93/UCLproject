%% Get the data 1 and 2(each has 256 points)
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==4);
label2 = find(labels==7);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:256);
label2 = label2(1:256);

totallabel = [label1;label2];

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

%% visualise the KNN graph 
%%red is label1 number 4
%%blue is label2 number 7

s = sum(A);
di = diag(sparse(1./s));
m = di*A;
[vector,eigenval] = eigs(m,6);
gplot(A,vector(:,2:3),'k');
hold on
gplot(A,vector(:,2:3),'bo');
gplot(A(1:256,1:256),vector(:,2:3),'ro');