load 'E:/semi-supervised/USPS/usps_all.mat';
datatouse = reshape(data, 256, 11000);
datadouble = double(datatouse);
datacompute = datadouble/255;

distance = zeros(11000,11000);
distance = distance + 10000;
for i=1:11000
    
    m = datacompute(:,i);
    repm = repmat(m,1,11000);
    dis = datacompute-repm;
    eachdis = dis.^2;
    totaldis = sum(eachdis,1);
    distance(i,:)=totaldis;
    fprintf('%d...\n', i);
   
end

save distance_usps_data distance;