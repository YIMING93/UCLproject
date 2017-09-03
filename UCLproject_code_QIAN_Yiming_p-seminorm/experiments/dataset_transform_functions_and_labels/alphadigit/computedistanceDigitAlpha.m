load 'E:/semi-supervised/USPS/binaryalphadigs.mat'
dataDA = dat;
dataDA = dataDA';
dataDA = reshape(dataDA, 1404, 1);

suitmat = zeros(20,16);
dataDAuse = zeros(320, 1404);
resuitmat = zeros(320,1);

for i=1:1404
    suitmat = dataDA{i};
    resuitmat = reshape(suitmat, 320,1);
    dataDAuse(:,i) = resuitmat;

end

dataDAforuse = double(dataDAuse);

distance = zeros(1404,1404);
distance = distance + 10000;
for i=1:1404
    
    m = dataDAforuse(:,i);
    repm = repmat(m,1,1404);
    dis = dataDAforuse-repm;
    eachdis = dis.^2;
    totaldis = sum(eachdis,1);
    distance(i,:)=totaldis;
    fprintf('%d...\n', i);
   
end

save distance_binaryalphadigs_data distance;





