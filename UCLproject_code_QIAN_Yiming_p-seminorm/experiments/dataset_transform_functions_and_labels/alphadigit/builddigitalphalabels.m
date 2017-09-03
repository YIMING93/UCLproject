load 'E:/semi-supervised/USPS/binaryalphadigs.mat'
dataDA = dat;
dataDA = dataDA';
dataDA = reshape(dataDA, 1404, 1);

class_labels = classlabels;
class_labels = repmat(class_labels, 39,1);
class_labels = char(class_labels);


save binaryalphadigs_labels class_labels;