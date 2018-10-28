% clear all
clc

double_flag = true;

if (double_flag)
    data_type = 'double';
else
    data_type = 'float';
end


Hg = H
Qg = Q

% 
% N = 27*27;
% 
% d = round(100*rand(N,1)); % The diagonal values
% t = round(triu(bsxfun(@min,d,d.').*rand(N),1)); % The upper trianglar random values
% M = diag(d)+t+t.';


[m ~] = size(Hg)
[n ~] = size(Qg)


issymmetric(Hg) % Debe ser uno
[~,p] = chol(Qg) % p Debe ser zero definite


N = n;

% R = M;
% disp('solve')
% tic
% Rsol= eig(R);
% toc

% disp('eig done')

str_1 = strcat({'\nstatic'},{ '  '},{data_type});

R = Hg;
fid = fopen( 'acoustic_matrices.h', 'wt' );
fprintf(fid, str_1{1});
fprintf( fid,' H[%d] = {',N*N);
for i=1:N
    for j=1:N
        fprintf( fid,'%f,\t',R(i,j));
    end
    fprintf( fid,'\n');
end
fprintf( fid,'};\n');


R = Qg;
fprintf(fid, str_1{1});
fprintf( fid,' Q[%d] = {',N*N);
for i=1:N
    for j=1:N
        fprintf( fid,'%f,\t',R(i,j));
    end
    fprintf( fid,'\n');
end
fprintf( fid,'};\n');

% Rsol = LAMBDA;
% fprintf( fid,'\nstatic float lambda[%d] = {',N);
% for i=1:N
%     fprintf( fid,'%f,\t',Rsol(i));
% end
% fprintf( fid,'};\n');


fclose(fid);



