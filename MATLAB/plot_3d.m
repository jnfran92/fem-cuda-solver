

n = 29

P = zeros(n^3,3);
counter = 1;
for k=1:n
    for j=1:n
        for i=1:n
            P(counter,:) = [i,j,k];
            counter = counter + 1;
        end
    end
end

% % scale
% P = (P-1)/15;
% P(:,1) = P(:,1)*Lx;
% P(:,2) = P(:,2)*Ly;
% P(:,3) = P(:,3)*Lz;


% n_freq = 18
% C = A(:,n_freq);
set(0,'DefaultAxesColor',[1 1 1])
figure()

r = 3
t = 3
f = 5

for u=1:r*t
n_freq = u*f -1 ;
C = EigenVectors(:,n_freq);

subplot(r,t,u)
% set(0,'DefaultAxesColor',[1 1 1])
scatter3(P(:,1),P(:,2),P(:,3),13,C,'filled')
view(40,35)
colormap('jet')
% colorbar
axis off

title(['Mode # ',num2str(n_freq)],'fontsize',14)
% set(0,'DefaultAxesColor',[1 1 1])
end

% 
% 
% % [X,Y,Z] = sphere(16);
% [X,Y,Z] = cylinder(16);
% x = [0.5*X(:); 0.75*X(:); X(:)];
% y = [0.5*Y(:); 0.75*Y(:); Y(:)];
% z = [0.5*Z(:); 0.75*Z(:); Z(:)];
% 
% S = repmat([50,25,10],numel(X),1);
% C = repmat([1,2,3],numel(X),1);
% s = S(:);
% c = C(:);
% 
% figure
% scatter3(x,y,z,s,c)
% view(40,35)
% 
% 
% figure
% scatter3(X(:),Y(:),Z(:))
% view(40,35)
% 

