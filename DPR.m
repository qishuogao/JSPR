function [pre err] = DPR(p,GI,nList,lambda,Kmax,PreProcess)

% Relaxation with Edge Preserving

[K,no_samples] = size(p);
g = reshape(GI,[1 no_samples]);
pre = p;

for mmiter=1:Kmax
    pre_t1 = pre;
    for num_sample = 1:no_samples
        num = 0;
        den = 0;
        ui = pre(:,num_sample);
        pi = p(:,num_sample);
        numList = nList(num_sample,:);
        numList(numList== 0) =[] ;
        dem = sum(g(numList));
        for k=1:K
            num = sum(pre(k,numList).*g(numList));
            ukest(k) = ((1-lambda)*pi(k)+lambda*num)/((1-lambda)+lambda*dem+eps);
        end
        pre(:,num_sample) = ukest;
    end
    if PreProcess ~= 1
        pre = pre./repmat(sum(pre),K,1);
    end
    pre_t2 = pre;
    err(mmiter) = norm(pre_t2-pre_t1,1)/norm(pre_t1,1);
end
