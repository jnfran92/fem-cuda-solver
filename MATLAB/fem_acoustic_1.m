
clear
% clc

dofs_element = 8;
n = 4;
e = n - 1;


% n_total = n^3;

Lx = 0.414;
Ly = 0.314;
Lz = 0.360;

% rho_0 = 1.21;
c = 343;


a = [Lx,Ly,Lz]/(2*e);


B = [1 -1 -1 -1  1  1  1 -1;...
     1  1 -1 -1 -1 -1  1  1;...
     1  1  1 -1  1 -1 -1 -1;...
     1 -1  1 -1 -1  1 -1  1;...
     1 -1 -1  1  1 -1 -1  1;...
     1  1 -1  1 -1  1 -1 -1;...
     1  1  1  1  1  1  1  1 ;...
     1 -1  1  1 -1 -1  1 -1];
 

h = diag([0,...
     8/a(1)^2,...
     8/a(2)^2,...
     8/a(3)^2,...
     (8/3)*(1/a(1)^2 + 1/a(2)^2),...
     (8/3)*(1/a(1)^2 + 1/a(3)^2),...
     (8/3)*(1/a(2)^2 + 1/a(3)^2),...
     (8/9)*(1/a(1)^2 + 1/a(2)^2 + 1/a(3)^2)])...
     *a(1)*a(2)*(3) ;



q = diag([8,...
     8/3,...
     8/3,...
     8/3,...
     (8/9),...
     (8/9),...
     (8/9),...
     (8/27)])...
     *a(1)*a(2)*(3) ;

 

B_inv = inv(B);
B_inv_transp = B_inv';
 
H = B_inv_transp * h * B_inv;
Q = (1/c^2)* B_inv_transp * q * B_inv;

H = H * 1E10;
Q = Q * 1E10;


Hg = double(assemby_global(H,n));
Qg = double(assemby_global(Q,n));


% Hg = single(assemby_global(H,n));
% Qg = single(assemby_global(Q,n));


tic
% C?lculos de los vectores y valores propios
[A,LAMBDAI] = eig(Hg,Qg); % Kg,Mg
toc
LAMBDA = LAMBDAI*ones(length(LAMBDAI),1);


% Frecuencias naturales del sistema
Freqs = round(( LAMBDA.^(0.5) )./(2*pi));
% Se imprimen en consola las frecuencias
disp('15 Frecuencias naturales')
disp(Freqs(1:30))


% % Frecuencias naturales del sistema
% Freqs = round(( LAMBDA.^(0.5) )./(2*pi));
% % Se imprimen en consola las frecuencias
% disp('15 Frecuencias naturales')
% disp(Freqs(1:30))


% 
% % C?lculos de los vectores y valores propios
% [A,LAMBDAI] = eig(Hg,Qg); % Kg,Mg
% toc
% LAMBDA = LAMBDAI*ones(length(LAMBDAI),1);
% 
% 
% % Frecuencias naturales del sistema
% Freqs = round(( LAMBDA.^(0.5) )./(2*pi));
% % Se imprimen en consola las frecuencias
% disp('15 Frecuencias naturales')
% disp(Freqs(1:30))
% 
% 
% 
% 
% 
% % C?lculos de los vectores y valores propios
% [A,LAMBDAI] = eig(Hg(1:10,1:10),Qg(1:10,1:10)); % Kg,Mg
% toc
% LAMBDA = LAMBDAI*ones(length(LAMBDAI),1);
% 
% 
% % Frecuencias naturales del sistema
% Freqs = round(( LAMBDA.^(0.5) )./(2*pi));
% % Se imprimen en consola las frecuencias
% disp('15 Frecuencias naturales')
% disp(Freqs(1:end))
% 



% 
% Hg_Qg = Hg/Qg;
% % C?lculos de los vectores y valores propios
% [A,LAMBDAI] = eig(Hg_Qg); % Kg,Mg
% LAMBDA = LAMBDAI*ones(length(LAMBDAI),1);
% 
% 
% % Frecuencias naturales del sistema
% Freqs = round(( LAMBDA.^(0.5) )./(2*pi));
% % Se imprimen en consola las frecuencias
% disp('15 Frecuencias naturales')
% Freqs = sort(Freqs);
% disp(Freqs(1:15))




