#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

int n, m;
double* f;
double* b;
double* xstart;
double* xnext;
double* gradientf;
double* gradientg;
double* alpha;
double  lambda;

int check_point(double* x, int iter, double** g){
    int i, j, k, r, rezult = 1;
    double sum = 0.0;

    for(r = 0; r < m; r++){
        sum = 0.0;

        for(i = 0; i < n; i++)
            sum += g[r][i]*x[i];

        for(i = 0, k = n; i < n; i++){
            sum += g[r][k]*pow(x[i],2);
            for(j = i+1, k++; j < n; j++, k++)
                sum += g[r][k]*x[i]*x[j];
        }

        if(sum >= b[r]) alpha[r] = 0;
        else {
            alpha[r] = 0.001;
            rezult = 0;
        }
    }
    return rezult;
}


void find_gradient_g(double* x , int iter, double** g){
    int i, j, k, r;

    for(i = 0; i < n; i++)
        gradientg[i] = 0.0;

    for(r = 0; r < m; r++){
        for(i = 0; i < n; i++){
            gradientg[i] += alpha[r]*g[r][i];
            k = n;
            for(j = 0; j < i; k+=n-j, j++)
                gradientg[i] += alpha[r]*g[r][k+i-j]*x[j];

            gradientg[i] += alpha[r]*2*g[r][k]*x[i];

            for(j = i+1, k++; j < n; j++, k++)
                gradientg[i] += alpha[r]*g[r][k]*x[j];
        }
    }
}

void find_gradient_f(double* x){
    int i, j, k;

    for(i = 0; i < n; i++)
        gradientf[i] = 0.0;

    for(i = 0; i < n; i++){
        gradientf[i] += f[i];
        k = n;
        for(j = 0; j < i; k+=n-j, j++)
            gradientf[i] += f[k+i-j]*x[j];

        gradientf[i] += 2*f[k]*x[i];

        for(j = i+1, k++; j < n; j++, k++)
            gradientf[i] += f[k]*x[j];
    }
}

int check_gradient_f(){
    int i;

    for(i = 0; i < n; i++)
        if (gradientf[i] != 0) return 0;

    return 1;
}

double find_f(double* x){
    int i, j, k;
    double rezult = 0.0;

    for(i = 0; i < n; i++)
        rezult += f[i]*x[i];

    for(i = 0, k = n; i < n; i++){
        rezult += f[k]*pow(x[i],2);
        for(j = i+1, k++; j < n; j++, k++)
            rezult += f[k]*x[i]*x[j];
    }
    return rezult;
}

int find_ksi(int n1, int m1,  gsl_matrix * A, gsl_vector * B, gsl_vector * ksi, double** g){
    int i, j, k, r, t, c = 0, iter = 0, n2;
    double epsilon, fstart, fnext, temp;

    n = n1;
    m = m1;


    f = (double*)malloc((n+n*(n+1)/2)*sizeof(double));
    for(i = 0; i < n+n*(n+1)/2; i++){
        f[i] = 0;
    }

    f[n] = -1;
    j = n;

    for(i = n; i>=1; i-- ){
        j=j+i;
        f[j] = -1;
    }

    n2 = n1+n1*(n1+1)/2;
    b = (double*)malloc(m*sizeof(double));
    for(i = 0; i < m; i++){
       // printf("i = %d j = %d !!!!\n", i, j);
        for(j = 0; j < n+n*(n+1)/2; j++){
            if(j<n){

                g[i][j] = gsl_matrix_get (A, i, j);

            }else{

                g[i][j] = 0;

            }
        }
        b[i] = gsl_vector_get (B, i);

    }

    xstart = (double*)malloc(n*sizeof(double));
    xnext = (double*)malloc(n*sizeof(double));
    gradientf = (double*)malloc(n*sizeof(double));
    gradientg = (double*)malloc(n*sizeof(double));
    alpha = (double*)malloc(m*sizeof(double));

    epsilon = 0.01;
    lambda = 0.01;

    for(i = 0; i < n; i++){
        xstart[i] = 0.001;
    }

    //printf("X(%d) = (", iter);
    for(i = 0; i < n-1; i++)
        //printf("%7.3lf, ", xstart[i]);
   // printf("%7.3lf)\n", xstart[n-1]);

    fstart = find_f(xstart);
    //printf("f(x%d) = %7.3lf\n", iter, fstart);

    //printf("alpha(%d) = (", iter);
    for(i = 0; i < m-1; i++)
        //printf("%7.3lf, ", alpha[i]);
    //printf("%7.3lf)\n", alpha[m-1]);

    find_gradient_f(xstart);
   // printf("grad f(x%d) = (", iter);
    for(i = 0; i < n-1; i++)
        //printf("%7.3lf, ", gradientf[i]);
    //printf("%7.3lf)\n", gradientf[n-1]);

    find_gradient_g(xstart, iter, g);

    iter++;
    for(i = 0; i < n; i++){
        xnext[i] = xstart[i] + lambda*(gradientf[i]+gradientg[i]);
    }

    while(iter<3000){
        c = check_point(xnext, iter, g);

        if(c){
            fnext = find_f(xnext);
            if(fabs(fnext-fstart) < epsilon) break;
            fstart = fnext;
        }
        find_gradient_f(xnext);
        if(check_gradient_f()) break;

        find_gradient_g(xnext, iter, g);

        for(i = 0; i < n; i++)
            xstart[i] = xnext[i];

        iter++;
        for(i = 0; i < n; i++){
            xnext[i] = xstart[i] + lambda*(gradientf[i]+gradientg[i]);
        }

    }

    //printf("------------ОТВЕТ---------------------\n");
    //printf("X(%d) = (", iter);
    for(i = 0; i < n-1; i++)
        //printf("%f, ", xnext[i]);
    //printf("%f)\n", xnext[n-1]);
    //printf("f(x%d) = %f\n", iter, fnext);

    for(i = 0; i < n; i++){
        gsl_vector_set(ksi, i, xnext[i]);
    }
    return 0;
}