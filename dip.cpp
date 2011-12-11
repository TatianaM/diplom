#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "erroy.cpp"

int N_g;
int N_l;
int N_y;
int N_x;
int N_ineq;
int N_reg;
double ag, bg;

//C = A*B
int mult_matr_vect(int row, int column, gsl_matrix * A, gsl_vector * B, gsl_vector * C ){
    double t;
    int i, j;
    t = 0;
    for( i = 0; i < row; i++){
        for( j = 0; j < column; j++){
            t = gsl_matrix_get (A, i, j) * gsl_vector_get (B, j) + t;
        }

        gsl_vector_set(C, i, t);
        t = 0;
    }
    return 1;
}

//C = A*B
int mult_matr_diagmatr(int n,  gsl_matrix * A, gsl_vector * B, gsl_matrix * C ){
    double t;
    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            t = gsl_matrix_get (A, i, j);
            t = t*gsl_vector_get (B, j);
            gsl_matrix_set (C, i, j, t);
        }
    }
    return 1;
}

void matrix_print(int row, int column,  gsl_matrix * A){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < column; j++){
            printf ("%g ",   gsl_matrix_get (A, i, j));
        }
        printf ("\n");
    }
    printf ("\n\n\n");
}

void vector_print(int n,  gsl_vector * A){
    int i;
    for (i = 0; i < n; i++){
        printf ("%g ",   gsl_vector_get (A, i));
    }
    printf ("\n\n\n");
}

int init(gsl_matrix * f_exp,  gsl_vector * vector){
    FILE* fp;
    double t,  h, tay, dt, s2;
    int i, j;
    if((fp = fopen("C:/Workspace/dip/nu.txt", "rb")) == NULL){
        printf("file nu.txt is not found!!!!!!!!!!!!!!\n");
        exit(1);
    }

    //intagration limits
    fscanf(fp, "%lf", &ag);
    fscanf(fp, "%lf", &bg);

    h=bg/N_x;
    tay = 0;
    dt = ag;
    s2 = 16*3.14*3.14*1.4*1.4*5.3;
    for(i = 0; i < N_y; i++){
        fscanf(fp, "%lf", &t);
        gsl_vector_set (vector, i, t);
    }

    for(i = 0; i < N_y; i++){
        for(j = 0; j < N_x; j++){
            t = exp(-dt*tay*s2);
            gsl_matrix_set (f_exp, i, j, t);
            dt = dt + h;
        }

        //step in input function
        tay = tay + 0.00025;
        dt = ag;
    }
    return 1;
}


int  main (void){

    int i, j;
    double t, t1,  alpha_reg;

    //  regularization parametr!!!!
    alpha_reg =0.01555;

    //number of points!!!!!
    N_g = 24;
    N_l = 0;
    N_x = N_g + N_l;
    N_ineq = N_g;
    N_reg = N_g+2;
    N_y = N_x;

    gsl_vector *d = gsl_vector_calloc (N_ineq);
    gsl_matrix *D = gsl_matrix_calloc (N_ineq, N_x);
    gsl_vector *x = gsl_vector_calloc (N_x);
    gsl_matrix *A = gsl_matrix_calloc (N_y, N_x);
    gsl_matrix *R = gsl_matrix_calloc (N_reg, N_x);
    gsl_vector *r = gsl_vector_calloc (N_reg);
    gsl_matrix *K2 = gsl_matrix_calloc (N_x, N_x);

    gsl_matrix *U = gsl_matrix_alloc (N_reg, N_x);
    gsl_vector *H = gsl_vector_alloc (N_x);
    gsl_matrix *Z = gsl_matrix_alloc (N_x, N_x);

    gsl_vector *work = gsl_vector_alloc (N_x);
    gsl_matrix *M_e = gsl_matrix_calloc (N_y, N_y);
    gsl_matrix *C = gsl_matrix_calloc (N_y, N_x);
    gsl_vector *nu = gsl_vector_alloc (N_y);
    gsl_matrix *temp = gsl_matrix_alloc (N_y, N_x);
    gsl_matrix *ZH = gsl_matrix_alloc (N_x, N_x);

    gsl_matrix *Q = gsl_matrix_alloc (N_y, N_x);
    gsl_vector *S = gsl_vector_alloc (N_x);
    gsl_matrix *W = gsl_matrix_alloc (N_x, N_x);

    gsl_matrix *QT = gsl_matrix_alloc (N_x, N_y);
    gsl_vector *gamma = gsl_vector_alloc (N_x);
    gsl_matrix *WS = gsl_matrix_alloc (N_x, N_x);
    gsl_matrix *ZHWS = gsl_matrix_alloc (N_x, N_x);
    gsl_vector *temp_vect = gsl_vector_alloc (N_x);
    gsl_vector *ksi_ineq = gsl_vector_alloc (N_x);
    gsl_vector *ksi = gsl_vector_calloc (N_x);

    init(A, nu);
    //vector_print(N_y, nu);


    for(i  = 0; i < N_g; i++){
        gsl_matrix_set (R, i, i, 1.0);
        gsl_matrix_set (R, i+1, i, -2.0);
        gsl_matrix_set (R, i+2, i, 1.0);
    }

   // matrix_print(N_reg, N_x, R);

    gsl_matrix_set_identity(D);
    gsl_matrix_set_identity(K2);
    gsl_matrix_set_identity(M_e);
    gsl_matrix_memcpy (U, R);

    //(A.7)
    gsl_linalg_SV_decomp (U, Z, H, work);
    gsl_linalg_matmult (M_e, A, C);

    //find H^-1
    for(i = 0; i < N_x; i++){
        t = gsl_vector_get (H, i);
        t=1/t;
        gsl_vector_set (H, i, t);
    }

    //ZH = Z * H^-1
    mult_matr_diagmatr(N_x,  Z, H, ZH);
    gsl_linalg_matmult (C, ZH, temp);
    gsl_matrix_memcpy (Q, temp);

    //(A.15)
    gsl_linalg_SV_decomp (Q, W, S, work);
    gsl_matrix_transpose_memcpy (QT, Q);

    //gamma = Qt * nu (A.19)
    mult_matr_vect(N_x, N_y, QT, nu, gamma );

    //calculate gamma ~ and S~ ^(-1)   (A.21)  (A.22)
    for( i = 0; i < N_x; i++){
        t = gsl_vector_get(S, i);
        t1 = 1 / sqrt(t * t + alpha_reg * alpha_reg);
        t = gsl_vector_get(gamma, i) * t * t1;
        gsl_vector_set(gamma, i, t);
        gsl_vector_set(S, i, t1);
    }


    mult_matr_diagmatr(N_x,  W, S, WS);
    gsl_linalg_matmult (ZH, WS, ZHWS);
    mult_matr_vect(N_x, N_x, WS, gamma, temp_vect );
    mult_matr_vect(N_x, N_x, ZH, temp_vect, ksi_ineq );

    for(i = 0; i < N_x; i++){
        t = gsl_vector_get(ksi_ineq, i);
        gsl_vector_set(ksi_ineq, i, -t);
    }

   // printf ("!!!\n");
   //matrix_print(N_x, N_x, ZHWS);
   // vector_print(N_x, ksi_ineq);

    double** g;

    g = (double**)malloc(N_x*sizeof(double*));

    for(i = 0; i < N_x; i++){
        g[i]=(double*)malloc((N_x+N_x*(N_x+1)/2)*sizeof(double));
    }

    find_ksi(N_x, N_x, ZHWS, ksi_ineq, ksi, g);

    for(i = 0; i < N_x; i++){
        t = gsl_vector_get(ksi, i);
        t1 = gsl_vector_get(gamma, i);
        gsl_vector_set(ksi, i, t+t1);
    }

    mult_matr_vect(N_x, N_x, WS, ksi, temp_vect );
    mult_matr_vect(N_x, N_x, ZH, temp_vect, x );

   //vector_print(N_x, x);

    mult_matr_vect(N_x, N_x, A, x, ksi );
   // vector_print(N_x, ksi);

    double sum, tp, tn;
    sum = 0;
    double   h, tay, dt;

    for(i = 0; i < N_x-1; i++){
        t = gsl_vector_get(nu, i);
        tp = gsl_vector_get(nu, i+1);
        sum =(t+tp)/2;
       // printf("%lf\n", t);
       // printf("%lf\n", sum);
    }

    //printf(" sum = %lf\n", sum);

    mult_matr_vect(N_x, N_x, A, nu, ksi );
    //vector_print(N_x, ksi);
    t = log(exp(-900*0.001))*(-1000);
    printf("t = %lf\n", t);
    t = exp(-550*0.001);
    printf("t = %lf\n", t);

    h = 0.0005;
    dt = 0;

    FILE* fp;

    if((fp = fopen("C:/Workspace/dip/exper.data", "w")) == NULL){
        printf("file exper.data is not found!!!!!!!!!!!!!!!!\n");
        exit(1);
    }

    i = 0;
    for(dt = ag; i < N_x; dt = dt +(bg-ag)/N_x ){
        t = gsl_vector_get(x, i);
        //fprintf(fp, "%f %f\n", dt, t);
        //printf("!!%f %f\n", dt, t);
        i++;
    }


    for(dt = 0; dt < 0.007; dt = dt + 0.00015){
        t = exp(-800*dt);
        //printf("%f \n",  t);

    }

    t= 600*575/(1.3*8*3.14*2);
    //printf("x = %f \n", t);

    t= 1.4*1.4*sin(90*3.14/180)*sin(90*3.14/180)/142.1;
    printf("lambda = %f \n", sqrt(t));

    for(i = 1; i < N_x-1;i++){
        tp = gsl_vector_get(x, i-1);
        t = gsl_vector_get(x, i);
        tn = gsl_vector_get(x, i+1);
        if(t>tp && t> tn){
            t = ag+(bg-ag)/N_x*(i);
            printf("%d %f razmer = %f \n", i+1, t, 1.381*300*10/(t*6*M_PI));
        }
    }

    return 0;

}