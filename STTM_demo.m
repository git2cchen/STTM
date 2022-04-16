% STTM demo on cifar-10

clear all

load cifar_10_train
load cifar_10_test

class_num=2;
W_r=[ones(31,1),[2:32]',3*ones(31,1),ones(31,1)]; % the TT-ranks for hyperplane  
sample=[1000]';% sample number

for s=1:1
for loop=1:31
%% model training
traindata=[train_X(1:sample(s),:);train_X(5001:sample(s)+5000,:)];
trainingL=[zeros(sample(s),1);ones(sample(s),1)]; 
[e,labels,W,b]=f2_STTM(traindata,trainingL,W_r(loop,:));

%% validation
validdata=[train_X(sample(s)+1:sample(s)+1000,:);train_X(sample(s)+5001:sample(s)+6000,:)];
validL=[zeros(sample(s),1);ones(sample(s),1)]; 
X=validdata;
N=size(X,1);
X=reshape(X,[N 32 32 3]);
X=permute(X,[2 3 4 1]);%7*4*4*7*60000 tensor
X=tt_tensor(X,1e-2); %do not do truncation here first, but can use TT-feature to replace to save the storage and computation
valid_X=full(X,[3072 N]);
valid_X=valid_X';

W_new=full(cell2core(tt_tensor(1),W));
A=valid_X*W_new+b;
B=zeros(size(valid_X,1),1);
B(A<0)=0;B(A>0)=1;  %need to care here

diff=B-validL;
diff(diff~=0)=1;
error=sum(diff)/N;
valid_error(loop,s)=error;


end
end

[~,ind]=min(valid_error);
traindata=[train_X(1:sample(s),:);train_X(5001:sample(s)+5000,:)];
trainingL=[zeros(sample(s),1);ones(sample(s),1)]; 
[e,labels,W,b]=f2_STTM(traindata,trainingL,W_r(ind,:));
%% test
X=[test_X(1:1000,:);test_X(1001:2000,:)];
testingL=[zeros(1000,1);ones(1000,1)]; 
N=size(X,1);
X=reshape(X,[N 32 32 3]);
X=permute(X,[2 3 4 1]);%7*4*4*7*60000 tensor
X=tt_tensor(X,1e-2); %do not do truncation here first, but can use TT-feature to replace to save the storage and computation
test_X=full(X,[3072 N]);
test_X=test_X';

W_new=full(cell2core(tt_tensor(1),W));
A=test_X*W_new+b;
B=zeros(size(test_X,1),1);
B(A<0)=0;B(A>0)=1;  %need to care here

diff=B-testingL;
diff(diff~=0)=1;
error=sum(diff)/N;
test_error=error
