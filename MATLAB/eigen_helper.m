clear all
clc


N = 16*16*16;

d = round(100*rand(N,1)); % The diagonal values
t = round(triu(bsxfun(@min,d,d.').*rand(N),1)); % The upper trianglar random values
M = diag(d)+t+t.';


% n_eig_mat = 4;

% R = magic(N);
R = M;
disp('solve')
tic
Rsol= eig(R);
toc

disp('eig done')

fid = fopen( 'sparse_matrix.h', 'wt' );
fprintf( fid,'\nstatic double A[%d] = {',N*N);
for i=1:N
    for j=1:N
        fprintf( fid,'%f,\t',R(i,j));
    end
    fprintf( fid,'\n');
end
fprintf( fid,'};\n');
% fclose(fid);


fprintf( fid,'\nstatic double lambda[%d] = {',N);
for i=1:N
    fprintf( fid,'%f,\t',Rsol(i));
end
fprintf( fid,'};\n');
fclose(fid);



% dlmwrite('myFile.txt',Cell,'delimiter',',','precision','%.6f');

% 
% A  = rand(10,1);
% B = rand(10,1);
% header1 = 'Hello';
% header2 = 'World!';
% fid=fopen('My3File.txt','w');
% fprintf(fid, [ header1 ' ' header2 '\n']);
% fprintf(fid, '%f ,%f \n', [A B]');
% fclose(fid);



% 
% Rt = R;
% Rs = Rt(:);
% str_out = strcat('{',num2str(Rs(1)));
% counter = 0;
% for i=2:length(Rs)
%     str_out = strcat(str_out, ' ,',num2str(Rs(i)));
% %     disp(num2str(Rs(i)))
%     counter = counter +1;
% end
% str_out = strcat(str_out,'}');
% str_out = strcat('\ndouble A[',num2str(N*N),']=',str_out,';\n');
% toc()
% % disp(str_out)
% 
% fid = fopen( 'sparse_matrix.h', 'wt' );
% fprintf( fid, str_out);
% % fclose(fid);
% 
% 
% 
% 
% % Rt = Rsol;
% Rt = ones(1,N);
% Rs = Rt(:);
% str_out = strcat('{',num2str(Rs(1)));
% counter = 0;
% for i=2:length(Rs)
%     str_out = strcat(str_out, ' ,',num2str(Rs(i)));
% %     disp(num2str(Rs(i)))
%     counter = counter +1;
% end
% str_out = strcat(str_out,'}');
% str_out = strcat('double lambda[',num2str(N),']=',str_out,';\n');
% % disp(str_out)
% 
% 
% % fid = fopen( 'Lambda.h', 'wt' );
% fprintf( fid, str_out);
% fclose(fid);
% 
% % disp(num2str(Rs(i)))

