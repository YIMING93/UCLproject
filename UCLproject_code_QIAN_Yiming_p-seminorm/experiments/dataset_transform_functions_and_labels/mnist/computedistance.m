images = loadMNISTImages('E:/semi-supervised/t10k-images.idx3-ubyte');
distance = zeros(10000,10000);
distance = distance + 1000;
for i=1:10000
    
    m = images(:,i);
    repm = repmat(m,1,10000);
    dis = images-repm;
    eachdis = dis.^2;
    totaldis = sum(eachdis,1);
    distance(i,:)=totaldis;
    fprintf('%d...\n', i);
   
end

save distance_data_t10k-images-idx3 distance;