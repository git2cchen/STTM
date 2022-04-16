%% for modification 

function [e, labels,W,b]=f2_STTM(train_X,train_labels,W_r);

X=train_X;
N=size(X,1);
X=reshape(X,[N 32 32 3]);
d=3;
X=permute(X,[2 3 4 1]);%32*32*3*N tensor
X=reshape(X,[size(X,1) size(X,2) size(X,3)*size(X,4)]);
X=tt_tensor(X,1e-2); %do not do truncation here first, but can use TT-feature to replace to save the storage and computation
X=core2cell(X);
X{d}=reshape(X{d},[size(X{d},1) size(X{d},2)/N N]); %it is a faster way to finish the procedure


% initialize my tensor train form W given the TT ranks W_r
% W_r=[1 5 3 1]; %[1 3 3 3 1] not suitable for many digit combination
d=length(W_r)-1;
for i=1:length(X) %get the feature dimension, namely 7 4 4 7
    fd(i)=size(X{i},2);
end
for i=1:length(X)-1 %get the X rank dimension
    Xd(i)=size(X{i},3);
end
Xd=[1 Xd N];

W=cell(1,d);

%% initialization method
W{1}=rand(W_r(1),fd(1),W_r(2));
W{1}=W{1}./norm(W{1}(:));
for i=d:-1:2
    W{i}=reshape(orth(rand(W_r(i)*fd(i),W_r(i+1))),[W_r(i),fd(i),W_r(i+1)]);
end
%
% W{1}=1*ones(W_r(1),fd(1),W_r(2));
% W{1}=W{1}./norm(W{1}(:));
% for i=d:-1:2
%     W{i}=1*ones(W_r(i),fd(i),W_r(i+1));
%     W{i}=W{i}./norm(W{i}(:));
% end

% SVMModel = fitcsvm(train_X,train_labels,'KernelFunction','linear',...
%           'ClassNames',{'0','1'},'OutlierFraction',0.02);
% w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1);
% w=reshape(w,[fd(1),prod(fd)/fd(1)]);
% for i=1:d-2
%     [U S V]=svd(w,'econ');
%      W{i}=reshape(U(:,1:W_r(i+1)),[W_r(i) fd(i) W_r(i+1)]);
%      w=reshape(S(1:W_r(i+1),1:W_r(i+1))*V(:,1:W_r(i+1))',[W_r(i+1)*fd(i+1) prod(fd(i+2:d))]);
% end
% [U S V]=svd(w,'econ');
% W{d-1}=reshape(U(:,1:W_r(d)),[W_r(d-1) fd(d-1) W_r(d)]);
% W{d}=S(1:W_r(d),1:W_r(d))*V(:,1:W_r(d))';
% % % W{1}=reshape(W{1},[1 size(W{1},1) size(W{1},2)]);

% clear w U S V
%%
% update the W cores one by one, untill small batch test data's classification
% error e is is less than a given threshold or max iteration
position=1;
ltr=1;
ite=1;
diff=1;
maxite=6; % an iteration is from 1~d than d~1

e(1)=1;

%% do cano here
for i=d:-1:position+1
    [Q,R]=qr(reshape(W{i},[W_r(i),fd(i)*W_r(i+1)])');
    W{i}=reshape(Q(:,1:W_r(i))',[W_r(i),fd(i),W_r(i+1)]);
    W{i-1}=reshape(reshape(W{i-1},[W_r(i-1)*fd(i-1),W_r(i)])*R(1:W_r(i),:)',[W_r(i-1),fd(i-1),W_r(i)]);
end

while (ite<2) ||((e(ite) < e(ite-1))&&(ite<maxite)&&(e(ite)>0.04))
    
    
    
    %  compute new input x
    if position==1
        
        
        X_t=reshape(full(cell2core(tt_tensor(1),X)),[fd(1) prod(fd(2:end)) N]);
        W_t=full(cell2core(tt_tensor(1),W(2:end)));
        x=reshape(tmprod(X_t,W_t,2),[fd(1)*W_r(2) N]);
        x=permute(x,[2 1]);%  [N fd(1)*W_r(2)]
        
        %train a svm by standard method
        SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
            'ClassNames',{'0','1'},'OutlierFraction',0.02);%,'OptimizeHyperParameters','auto'?%'solver','L1QP'?You cannot use QP for robust learning.
        
        %         SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
        %         'ClassNames',{'0','1'},'Solver','L1QP');
        %update the TT core
        %         w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1)/SVMModel.KernelParameters.Scale^2;
        w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1);
        W{position}=reshape(w,[1 fd(1) W_r(2)]);
    end
    
    if position==d
        X_t=reshape(full(cell2core(tt_tensor(1),X)),[prod(fd(1:end-1)) fd(end) N]);
        W_t=full(cell2core(tt_tensor(1),W(1:end-1)));
        x=reshape(tmprod(X_t,W_t',1),[W_r(end-1) fd(end) N]);
        x=reshape(permute(x,[3 2 1]),[N fd(end)*W_r(end-1)]);%  [N fd(end)*W_r(end-1)]
        
        %train a svm by standard method
        SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
            'ClassNames',{'0','1'},'OutlierFraction',0.02);%'solver','L1QP'?You cannot use QP for robust learning.
        
        %            SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
        %         'ClassNames',{'0','1'},'Solver','L1QP');
        %update the TT core
        %         w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1)/SVMModel.KernelParameters.Scale^2;
        w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1);
        W{position}=reshape(w,[ W_r(end-1) fd(end) 1]);
    end
    
    if (position~=1)&&(position~=d)
        X_tleft=reshape(full(cell2core(tt_tensor(1),X(1:position-1))),[prod(fd(1:position-1)) Xd(position)]);
        X_tright=reshape(full(cell2core(tt_tensor(1),X(position+1:end))),[Xd(position+1) prod(fd(position+1:end)) N]);
        W_tleft=full(cell2core(tt_tensor(1),W(1:position-1))); %prod(fd(1:position-1))*W_r(position-1)
        W_tright=full(cell2core(tt_tensor(1),W(position+1:end))); %W_r(position+1)* prod(fd(position+1:end))
        left=reshape(tmprod(X_tleft,W_tleft',1),[1 W_r(position) Xd(position)]);
        right=tmprod(X_tright,W_tright,2);
        U{1}=left;
        U{2}=X{position};
        U{3}=right;
        x=full(cell2core(tt_tensor(1),U));%W_r(position)*fd(position)*W_r(position+1)  *   N
        x=permute(x,[2 1]);% N  *  W_r(position)*fd(position)*W_r(position+1)
        
        %train a svm by standard method
        SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
            'ClassNames',{'0','1'},'OutlierFraction',0.02);%'solver','L1QP'?You cannot use QP for robust learning.
        
        %                 SVMModel = fitcsvm(x,train_labels,'KernelFunction','linear',...
        %         'ClassNames',{'0','1'},'Solver','L1QP');
        %update the TT core
        %         w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1)/SVMModel.KernelParameters.Scale^2;
        w=sum(SVMModel.Alpha.*SVMModel.SupportVectorLabels.*SVMModel.SupportVectors,1);
        W{position}=reshape(w,[W_r(position) fd(position) W_r(position+1)]);
    end
    
    if ltr
        position=position+1;
        if position== d
            ltr=0;
        end
    else
        position=position-1;
        if position== 1
            ltr=1;
        end
    end
    
    %     if (position==d) || (position==1) % half a sweep
    if (position==1)||(position==d)
        ite=ite+1;
        yhat=zeros(N,1);
        y=x*w'+SVMModel.Bias;
        yhat(y>0)=1;
        yhat(y<0)=0;
        
        diff=yhat-train_labels;
        diff(diff~=0)=1;
        e(ite)=sum(diff)/N;
        
        %         e(ite)=norm(yhat-train_labels)/norm(train_labels);
        %         diff=norm(full(cell2core(tt_tensor(1),Wf))-full(cell2core(tt_tensor(1),W)))/norm(full(cell2core(tt_tensor(1),Wf)))
        bf=SVMModel.Bias;
        Wf=W;% record the former W, will compute the difference later
        labels=yhat;
        %         scale=SVMModel.KernelParameters.Scale;
    end
    
    
    %% do not do normalization
    
    if (ltr) && (position~=1)
        [Q,R]=qr(reshape(W{position-1},[W_r(position-1)*fd(position-1),W_r(position-1+1)]));
        W{position-1}=reshape(Q(:,1:W_r(position-1+1)),[W_r(position-1),fd(position-1),W_r(position-1+1)]);
        W{position}=reshape(R(1:W_r(position),:)*reshape(W{position},[W_r(position),fd(position)*W_r(position+1)]),[W_r(position),fd(position),W_r(position+1)]);
        
    elseif (ltr) && (position==1)
        [Q,R]=qr(reshape(W{position+1},[W_r(position+1),fd(position+1)*W_r(position+1+1)])');
        W{position+1}=reshape(Q(:,1:W_r(position+1))',[W_r(position+1),fd(position+1),W_r(position+1+1)]);
        W{position}=reshape(reshape(W{position},[W_r(position)*fd(position),W_r(position+1)])*R(1:W_r(position+1),:)',[W_r(position),fd(position),W_r(position+1)]);
    elseif (~ltr) && (position~=d)
        [Q,R]=qr(reshape(W{position+1},[W_r(position+1),fd(position+1)*W_r(position+1+1)])');
        W{position+1}=reshape(Q(:,1:W_r(position+1))',[W_r(position+1),fd(position+1),W_r(position+1+1)]);
        W{position}=reshape(reshape(W{position},[W_r(position)*fd(position),W_r(position+1)])*R(1:W_r(position+1),:)',[W_r(position),fd(position),W_r(position+1)]);
    elseif (~ltr) && (position==d)
        [Q,R]=qr(reshape(W{position-1},[W_r(position-1)*fd(position-1),W_r(position-1+1)]));
        W{position-1}=reshape(Q(:,1:W_r(position-1+1)),[W_r(position-1),fd(position-1),W_r(position-1+1)]);
        W{position}=reshape(R(1:W_r(position),:)*reshape(W{position},[W_r(position),fd(position)*W_r(position+1)]),[W_r(position),fd(position),W_r(position+1)]);
    end
    
    
end


%%
W=Wf;
b=bf;


end