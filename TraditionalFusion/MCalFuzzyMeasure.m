function [mu D DD] = MCalFuzzyMeasure(score_data, label)
%% CalFuzzyMeasure - determine fuzzy measure from learning data
% 
% SYNTAX:
% [mu] = CalFuzzyMeasure(score_data, label)
%
% INPUT
% score_data - N*M*K matrix. The column of each N*M submatrix is 
%              an array containing confidence values given by N 
%              classifiers. K is the number of classes
%
% label      - class labels
%
% OUTPUTS:
% mu         - K *(2^n - 2)  values for fuzzy measure

%% currently only support binary class problem
[N, M, K] = size(score_data);
%assert( K== 2 );
%assert( length(label) == M );
%% generate subset constrain
P =  2^N - 1;
C = [];

for i = 1 : K*(P - 1)
    for j = i+1 : K*(P - 1)
        t = bitxor(i, j);
        t1 = bitand(i, t);
        t2 = bitand(j, t);
        if (t1 == 0 && t2 ~= 0)
            c = zeros(1, K*(P-1) );
            c(i) = 1;   % m_i - m_j <= 0
            c(j) = -1;
            C = [C; c];
        else if (t1 ~= 0 && t2 == 0)
                c = zeros(1, K*(P-1));
                c(i) = -1;  % m_j - m_i <= 0
                c(j) = 1;
                C = [C; c];
            end;
        end;
    end;
end;

b = [zeros(size(C, 1), 1); ones(K*(P-1), 1)];
C = [C; eye(K*(P-1))];

%% prepare for quadratic programming
[f, index] = sort(score_data, 'descend');
index = uint16(index);
D = zeros(K*P, M);
base = 0;
for k=1:K
    for i = 1 : M
        s = base;
        for j = 1 : N-1
            s = s + bitshift(1, index(j, i, k)-1);
            D(s, i) = f(j, i, k) - f(j+1, i, k);
        end;
        s = s + bitshift(1, index(N, i, k)-1);
        D(s, i) = f(N, i, k);
    end;
    base = base + P;
end;
DD = D;

z=label';
% size(D)
% size(z)
for k=1:K
    z(k,:)=D(k*P,:)-z(k,:);
end;
z=z';

for k=1:K
D(k*P-k+1, :) = [];
end;
H = D * D'+0.01*eye(size(D,1));
% cofficient=-ones(K*(P-1),K*(P-1))+2*eye(K*(P-1));
% H=H.*cofficient;

F=zeros(K*(P-1),1);
for k=0:K-1
   % F((k*(P-1)+1):(k+1)*(P-1),:)=F((k*(P-1)+1):(k+1)*(P-1),:)*label(k+1,:);
   %temp=D((k*(P-1)+1):(k+1)*(P-1),:)*label(:,k+1);
   temp=D((k*(P-1)+1):(k+1)*(P-1),:)*z(:,k+1);
   F((k*(P-1)+1):(k+1)*(P-1),:)=temp;
end;

%% quadratic programming
% [mu, lambda, how] = qpdantz(H, F, C, b, 0.1*ones(K*(P-1), 1), 100000);
[mu, lambda, how] = quadprog(H, F, C, b, [],[],0.1*ones(K*(P-1), 1));