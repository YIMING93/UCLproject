rand('seed',0);
%%Get the data 1 and 2(each has 128 points) the example codes
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

label1 = find(labels==1);
label2 = find(labels==2);

%% the total length of label1 is 1135, the total length of label2 is 1032
label1 = label1(1:128);
label2 = label2(1:128);

%% 
%% 
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

%% suppose label1 is 1 in first list and 0 for the second list 
%% suppose label2 is 0 in first list and 1 for the second list
%% column 1 is for label1 and column 2 is for label2
fl=zeros(8,2);
fl(1:4,1)=1;
fl(5:8,2)=1;

%% using harmonic function to do the compute
[fu, fu_CMN] = harmonic_function(A, fl);

fu = fu>0.5;

%% the true labels for unlabeled points
futrue = zeros(248,2);
futrue(1:124,1)=1;
futrue(125:248,2)=1;

%% compare the accuracy
final = (futrue(:,1)==fu(:,1));
accuracy = sum(final)/length(final);
fprintf('The accuracy is %.3f\n', accuracy);
%%fprintf('The accuracy is %d\n', sum(final));
%%fprintf('The accuracy is %.2f\n', length(final));












