function p = dof_n(i,j,k,n)
    p = i + n*(j-1) + n*n*(k-1);