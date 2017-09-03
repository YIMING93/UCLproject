%%Get the data 1 and 2
rand('seed',0);
load distance_data_t10k-images-idx3;
labels = loadMNISTLabels('E:/semi-supervised/t10k-labels.idx1-ubyte');

labelone = find(labels==1);
labeltwo = find(labels==2);



labelsample=[4,8,16,32];

pointssample=[64, 128, 256, 512, 1024];



for totalpoints=pointssample
eachpoints=totalpoints/2;
fprintf('\n\n\n\n\n\nThe total points is %d \n', totalpoints);


for labelsize=labelsample

eachsize = labelsize/2;

for K=3:7



%% the total length of labelone is 1135, the total length of labeltwo is 1032
label1 = labelone(1:eachpoints);
label2 = labeltwo(1:eachpoints);

%% 
%% 
accuracy = zeros(1,10);

%%run ten times for each case
for times=1:10
label1=label1(randperm(numel(label1)));
label2=label2(randperm(numel(label2)));

totallabel = [label1(1:eachsize);label2(1:eachsize);label1((eachsize+1):eachpoints);label2((eachsize+1):eachpoints)];

usedistance = distance(totallabel,totallabel);

%% build the KNN graphs (Using K) 

%% find the nearest two points(not connect by self)

usedistance1 = usedistance;
for i=1:totalpoints
    usedistance1(i,i)=1000000;
end

indexfornearest = zeros(totalpoints,K);

for i=1:totalpoints
    [Dis, indexfor] = sort(usedistance1(i,:));
    indexfornearest(i,:) = indexfor(1:K);
end

%% buide the unweight matrix A
A=zeros(totalpoints,totalpoints);
for i=1:totalpoints
    A(i,indexfornearest(i,:))=1;
    A(indexfornearest(i,:),i)=1;
end

%% suppose label1 is 1 in first list and 0 for the second list 
%% suppose label2 is 0 in first list and 1 for the second list
%% column 1 is for label1 and column 2 is for label2
fl=zeros(labelsize,2);
fl(1:eachsize,1)=1;
fl((eachsize+1):labelsize,2)=1;

%% using harmonic function to do the compute
[fu, fu_CMN] = harmonic_function(A, fl);

fu = fu>0.5;

%% the true labels for unlabeled points
countsize = totalpoints-labelsize;
halfcount = countsize/2;
futrue = zeros(countsize,2);
futrue(1:halfcount,1)=1;
futrue((halfcount+1):countsize,2)=1;

%% compare the accuracy
final = (futrue(:,1)==fu(:,1));
accuracy(times) = sum(final)/length(final);
%fprintf('The accuracy is %.3f  The K is %d \n', accuracy(times),K);
end

averageacc=sum(accuracy)/10;



%fprintf('The ten times average accuracy is %.3f  The K is %d  \n', averageacc,K);
fprintf('The ten times average accuracy is %.3f  The K is %d  The total label points number is %d\n', averageacc,K,labelsize);
%%fprintf('The accuracy is %d\n', sum(final));
%%fprintf('The accuracy is %.2f\n', length(final));

end

end

end