
#include <math.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <random>
//#include "SQuaD_MDS.h"

double standard_deviation_1D(double* flat_array, int N_entries) {
    double sum = 0;
    double sq_sum = 0;
    for (int i = 0; i < N_entries; i++) {
        sum += flat_array[i];
        sq_sum += flat_array[i] * flat_array[i];
    }
    double mean = sum / N_entries;
    double variance = sq_sum / N_entries - mean * mean;
    return sqrt(variance);
}

void init_embedding(double* flat_array, int N_entries, double target_stdev) {
    double std = standard_deviation_1D(flat_array, N_entries);
    double multiplier = target_stdev / std;
    for (int i = 0; i < N_entries; i++) {
        flat_array[i] *= multiplier;
    }
}


void fill_quartet_dists(int M, double* Xhd, int p1_hd_idx, int p2_hd_idx, int p3_hd_idx, int p4_hd_idx, double* Dhd_quartet,
                        double* tmp_params, int p1_ld_idx, int p2_ld_idx, int p3_ld_idx, int p4_ld_idx, double* Dld_quartet, bool squared_dists) {
    // HD temp variables
    double acc1 = 0.0,  acc2 = 0.0,  acc3 = 0.0,  acc4 = 0.0,  acc5 = 0.0,  acc6 = 0.0, quartet_sumdist;
    double diff1, diff2, diff3, diff4, diff5, diff6;
    // LD temp variables
    double acc1_LD = 0.0,  acc2_LD = 0.0,  acc3_LD = 0.0,  acc4_LD = 0.0,  acc5_LD = 0.0,  acc6_LD = 0.0, quartet_sumdist_LD;
    double diff1_LD, diff2_LD, diff3_LD, diff4_LD, diff5_LD, diff6_LD;
    
    // ---------------------------------- HD relative distances ----------------------------------
    //  accX will contain Euclidean^2 
    for (int dimension = 0; dimension < M; dimension++) {
        diff1 = Xhd[p1_hd_idx + dimension] - Xhd[p2_hd_idx + dimension];
        acc1 += diff1 * diff1;

        diff2 = Xhd[p1_hd_idx + dimension] - Xhd[p3_hd_idx + dimension];
        acc2 += diff2 * diff2;

        diff3 = Xhd[p1_hd_idx + dimension] - Xhd[p4_hd_idx + dimension];
        acc3 += diff3 * diff3;

        diff4 = Xhd[p2_hd_idx + dimension] - Xhd[p3_hd_idx + dimension];
        acc4 += diff4 * diff4;

        diff5 = Xhd[p2_hd_idx + dimension] - Xhd[p4_hd_idx + dimension];
        acc5 += diff5 * diff5;

        diff6 = Xhd[p3_hd_idx + dimension] - Xhd[p4_hd_idx + dimension];
        acc6 += diff6 * diff6;

    }
    
    // transform accX to real Euclidean
    if (!squared_dists){ 
        acc1 = sqrt(acc1);
        acc2 = sqrt(acc2);
        acc3 = sqrt(acc3);
        acc4 = sqrt(acc4);
        acc5 = sqrt(acc5);
        acc6 = sqrt(acc6);
    }
    
    // now the relaitve distances can be computed
    quartet_sumdist = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + 6*1e-12;
    Dhd_quartet[0] = acc1 / quartet_sumdist;
    Dhd_quartet[1] = acc2 / quartet_sumdist;
    Dhd_quartet[2] = acc3 / quartet_sumdist;
    Dhd_quartet[3] = acc4 / quartet_sumdist;
    Dhd_quartet[4] = acc5 / quartet_sumdist;
    Dhd_quartet[5] = acc6 / quartet_sumdist;


    // ---------------------------------- LD distances ----------------------------------
    for (int dimension = 0; dimension < 2; dimension++) { // TODO manual loop unrolling ??
        diff1_LD = tmp_params[p1_ld_idx + dimension] - tmp_params[p2_ld_idx + dimension];
        acc1_LD += diff1_LD * diff1_LD;

        diff2_LD = tmp_params[p1_ld_idx + dimension] - tmp_params[p3_ld_idx + dimension];
        acc2_LD += diff2_LD * diff2_LD;

        diff3_LD = tmp_params[p1_ld_idx + dimension] - tmp_params[p4_ld_idx + dimension];
        acc3_LD += diff3_LD * diff3_LD;

        diff4_LD = tmp_params[p2_ld_idx + dimension] - tmp_params[p3_ld_idx + dimension];
        acc4_LD += diff4_LD * diff4_LD;

        diff5_LD = tmp_params[p2_ld_idx + dimension] - tmp_params[p4_ld_idx + dimension];
        acc5_LD += diff5_LD * diff5_LD;

        diff6_LD = tmp_params[p3_ld_idx + dimension] - tmp_params[p4_ld_idx + dimension];
        acc6_LD += diff6_LD * diff6_LD;
    }
    acc1_LD = sqrt(acc1_LD) + 1e-12;
    acc2_LD = sqrt(acc2_LD) + 1e-12;
    acc3_LD = sqrt(acc3_LD) + 1e-12;
    acc4_LD = sqrt(acc4_LD) + 1e-12;
    acc5_LD = sqrt(acc5_LD) + 1e-12;
    acc6_LD = sqrt(acc6_LD) + 1e-12;
    
    quartet_sumdist_LD = acc1_LD + acc2_LD + acc3_LD + acc4_LD + acc5_LD + acc6_LD;
    Dld_quartet[0] = acc1_LD;
    Dld_quartet[1] = acc2_LD;
    Dld_quartet[2] = acc3_LD;
    Dld_quartet[3] = acc4_LD;
    Dld_quartet[4] = acc5_LD;
    Dld_quartet[5] = acc6_LD;

}

void ABCD_grad(double xa, double ya, double xb, double yb, double xc, double yc, double xd, double yd,
               double dab, double dac, double dad, double dbc, double dbd, double dcd, double sum_dist,
               double pab, double* quartet_grads){
    double ratio1 = dab/sum_dist;
    double direction = ((double)2)*((pab - ratio1)/sum_dist);
    
    quartet_grads[0] = direction * (ratio1 * ((xa-xb)/dab + (xa-xc)/dac + (xa-xd)/dad ) - (xa-xb)/dab );
    quartet_grads[1] = direction * (ratio1 * ((ya-yb)/dab + (ya-yc)/dac + (ya-yd)/dad ) - (ya-yb)/dab );

    quartet_grads[2] = direction * (ratio1 * ((xb-xa)/dab + (xb-xc)/dbc + (xb-xd)/dbd ) - (xb-xa)/dab );
    quartet_grads[3] = direction * (ratio1 * ((yb-ya)/dab + (yb-yc)/dbc + (yb-yd)/dbd ) - (yb-ya)/dab );

    quartet_grads[4] = direction * (ratio1 * ((xc-xa)/dac + (xc-xb)/dbc + (xc-xd)/dcd ));
    quartet_grads[5] = direction * (ratio1 * ((yc-ya)/dac + (yc-yb)/dbc + (yc-yd)/dcd ));

    quartet_grads[6] = direction * (ratio1 * ((xd-xa)/dad + (xd-xb)/dbd + (xd-xc)/dcd ));
    quartet_grads[7] = direction * (ratio1 * ((yd-ya)/dad + (yd-yb)/dbd + (yd-yc)/dcd ));
    
}


void fill_quartet_grads(double* quartet_grads, double* Dhd_quartet, double* Dld_quartet, double* tmp_params, int p1_ld_idx, int p2_ld_idx, int p3_ld_idx, int p4_ld_idx) {
    
    double xa = tmp_params[p1_ld_idx], ya = tmp_params[p1_ld_idx+1];
    double xb = tmp_params[p2_ld_idx], yb = tmp_params[p2_ld_idx+1];
    double xc = tmp_params[p3_ld_idx], yc = tmp_params[p3_ld_idx+1];
    double xd = tmp_params[p4_ld_idx], yd = tmp_params[p4_ld_idx+1];
    double d_ab = Dld_quartet[0];
    double d_ac = Dld_quartet[1];
    double d_ad = Dld_quartet[2];
    double d_bc = Dld_quartet[3];
    double d_bd = Dld_quartet[4];
    double d_cd = Dld_quartet[5];
    double sum_dist = d_ab + d_ac + d_ad + d_bc + d_bd + d_cd;
    
    
    double gxA = 0, gyA = 0, gxB = 0, gyB = 0, gxC = 0, gyC = 0, gxD = 0, gyD = 0;
    ABCD_grad(xa, ya, xb, yb, xc, yc, xd, yd, d_ab, d_ac, d_ad, d_bc, d_bd, d_cd, sum_dist, Dhd_quartet[0], quartet_grads); // gradient for distance between 1 and 2 (AB)
    gxA += quartet_grads[0];gyA += quartet_grads[1];
    gxB += quartet_grads[2];gyB += quartet_grads[3];
    gxC += quartet_grads[4];gyC += quartet_grads[5];
    gxD += quartet_grads[6];gyD += quartet_grads[7];
    ABCD_grad(xa, ya, xc, yc, xb, yb, xd, yd, d_ac, d_ab, d_ad, d_bc, d_cd, d_bd, sum_dist, Dhd_quartet[1], quartet_grads); // gradient for distance between 1 and 3 (AC)
    gxA += quartet_grads[0];gyA += quartet_grads[1];
    gxC += quartet_grads[2];gyC += quartet_grads[3];
    gxB += quartet_grads[4];gyB += quartet_grads[5];
    gxD += quartet_grads[6];gyD += quartet_grads[7];
    ABCD_grad(xa, ya, xd, yd, xc, yc, xb, yb, d_ad, d_ac, d_ab, d_cd, d_bd, d_bc, sum_dist, Dhd_quartet[2], quartet_grads); // gradient for distance between 1 and 4 (AD)
    gxA += quartet_grads[0];gyA += quartet_grads[1];
    gxD += quartet_grads[2];gyD += quartet_grads[3];
    gxC += quartet_grads[4];gyC += quartet_grads[5];
    gxB += quartet_grads[6];gyB += quartet_grads[7];
    ABCD_grad(xb, yb, xc, yc, xa, ya, xd, yd, d_bc, d_ab, d_bd, d_ac, d_cd, d_ad, sum_dist, Dhd_quartet[3], quartet_grads); // gradient for distance between 2 and 3 (BC)
    gxB += quartet_grads[0];gyB += quartet_grads[1];
    gxC += quartet_grads[2];gyC += quartet_grads[3];
    gxA += quartet_grads[4];gyA += quartet_grads[5];
    gxD += quartet_grads[6];gyD += quartet_grads[7];
    ABCD_grad(xb, yb, xd, yd, xa, ya, xc, yc, d_bd, d_ab, d_bc, d_ad, d_cd, d_ac, sum_dist, Dhd_quartet[4], quartet_grads); // gradient for distance between 2 and 4 (BD)
    gxB += quartet_grads[0];gyB += quartet_grads[1];
    gxD += quartet_grads[2];gyD += quartet_grads[3];
    gxA += quartet_grads[4];gyA += quartet_grads[5];
    gxC += quartet_grads[6];gyC += quartet_grads[7];
    ABCD_grad(xc, yc, xd, yd, xa, ya, xb, yb, d_cd, d_ac, d_bc, d_ad, d_bd, d_ab, sum_dist, Dhd_quartet[5], quartet_grads); // gradient for distance between 3 and 4 (CD)
    gxC += quartet_grads[0];gyC += quartet_grads[1];
    gxD += quartet_grads[2];gyD += quartet_grads[3];
    gxA += quartet_grads[4];gyA += quartet_grads[5];
    gxB += quartet_grads[6];gyB += quartet_grads[7];

    quartet_grads[0] = gxA;
    quartet_grads[1] = gyA;
    quartet_grads[2] = gxB;
    quartet_grads[3] = gyB;
    quartet_grads[4] = gxC;
    quartet_grads[5] = gyC;
    quartet_grads[6] = gxD;
    quartet_grads[7] = gyD;
}


void fill_MDS_grads(int N, int M, double* Xhd, double* tmp_params, double* grads,  double* quartet_grads, int* perms, double* Dhd_quartet, double* Dld_quartet,bool distance_exaggeration) {
    int p1, p2, p3, p4, batch_cursor;
    int p1_hd_idx, p2_hd_idx, p3_hd_idx, p4_hd_idx;
    int p1_ld_idx, p2_ld_idx, p3_ld_idx, p4_ld_idx;
    int N_quartets = (int) ((N-4) / 4) ;

    for (int batch_nb = 0; batch_nb < N_quartets; batch_nb++) {
        batch_cursor = batch_nb * 4;
        p1 = perms[batch_cursor];  p2 = perms[batch_cursor +1];  p3 = perms[batch_cursor +2];  p4 = perms[batch_cursor +3];
        p1_hd_idx = p1 * M; p2_hd_idx = p2 * M; p3_hd_idx = p3 * M; p4_hd_idx = p4 * M;
        p1_ld_idx = p1 * 2; p2_ld_idx = p2 * 2; p3_ld_idx = p3 * 2; p4_ld_idx = p4 * 2;
        
        // fill Dhd_quartet and Dld_quartet with LD and HD relative distances of the quartet
        fill_quartet_dists(M, Xhd, p1_hd_idx, p2_hd_idx, p3_hd_idx, p4_hd_idx, Dhd_quartet, tmp_params, p1_ld_idx, p2_ld_idx, p3_ld_idx, p4_ld_idx, Dld_quartet, distance_exaggeration);
               
        // the gradients for the quartet can now be computed and saved in quartet_grads
        fill_quartet_grads(quartet_grads, Dhd_quartet, Dld_quartet, tmp_params, p1_ld_idx, p2_ld_idx, p3_ld_idx, p4_ld_idx);

        // now, simply fill grads (contains all the grads) with the content of quartet_grads (contains the grads for 4 points) at the appropriate locations
        grads[p1_ld_idx]   = quartet_grads[0];
        grads[p1_ld_idx+1] = quartet_grads[1];
        grads[p2_ld_idx]   = quartet_grads[2];
        grads[p2_ld_idx+1] = quartet_grads[3];
        grads[p3_ld_idx]   = quartet_grads[4];
        grads[p3_ld_idx+1] = quartet_grads[5];
        grads[p4_ld_idx]   = quartet_grads[6];
        grads[p4_ld_idx+1] = quartet_grads[7];
        
    }
}


void update_momentum_and_Xld(double* grads, int N, double* Xld_flat, double* momentums, double LR){
    int x_idx, y_idx;

    double norm = 0.;
    for (int point_nb = 0; point_nb < N; point_nb++){
        x_idx = point_nb * 2;
        y_idx = x_idx + 1;
        norm  += grads[x_idx]*grads[x_idx] + grads[y_idx]*grads[y_idx];
    }
    norm = sqrt(norm);
    
    double multiplier = LR / norm; // remember that LR grows with N
    for (int point_nb = 0; point_nb < N; point_nb++){ // for each point in LD space ...
        x_idx = point_nb * 2;
        y_idx = x_idx + 1;
        
        // update momentum
        momentums[x_idx] -= multiplier * grads[x_idx];
        momentums[y_idx] -= multiplier * grads[y_idx];
        
        // update Xld
        Xld_flat[x_idx] += momentums[x_idx];
        Xld_flat[y_idx] += momentums[y_idx];
           
    }
}


void custom_nestrov_iteration(int N, int M, double* Xhd, double* Xld_flat, double* tmp_params, double* grads, double* momentums,int* perms, double* Dhd_quartet, double* Dld_quartet, double* quartet_grads,double LR, bool distance_exaggeration) {
    // apply rudimentary friction to momentum and zero the gradients. also, fill the tmp params.
    for (int param_idx = 0; param_idx < 2 * N; param_idx++) {
        momentums[param_idx] *= 0.99;  // tiny friction
        tmp_params[param_idx] = Xld_flat[param_idx] + momentums[param_idx]; // use these params for nestrov gradients
    }
    
    // fill the grads for each point in LD with SQuaD_MDS gradients
    fill_MDS_grads(N, M, Xhd, tmp_params, grads, quartet_grads, perms, Dhd_quartet, Dld_quartet, distance_exaggeration);

    // normalise the gradients, update momentum, and finaly update Xld
    update_momentum_and_Xld(grads, N, Xld_flat, momentums, LR);
}



int run_SQuaD_MDS(double* Xhd, int N, int M, double* Xld_flat, int n_iter) {
    init_embedding(Xld_flat, 2 * N, 10.0);
    
    // optimiser strategy params
    double LR_init = max(2., 0.005*(double)N);
    double LR = LR_init;
    int decay_start = (int) (0.1 * (double) n_iter); // it can be worth it to start the decay later, especially with random init
    bool distance_exaggeration = false;  // exagerate (=square) the quartet HD distances during the first couple of iterations. It can help, especially on random inits
    if (decay_start > 0)
        bool distance_exaggeration = true;
    
    double decay_cte = 0.34;
    //double decay_cte = 0.2;
    double decay_offset = -exp(-1 / decay_cte);
    double temp_ratio = 1.; // used for decay
    //double decay = exp(log(1e-4) / n_iter);
    
    double* momentums  = (double*) malloc(2 * N * sizeof(double));
    double* grads      = (double*) malloc(2 * N * sizeof(double));
    double* tmp_params = (double*) malloc(2 * N * sizeof(double));
    double* Dhd_quartet = (double*) malloc(6 * sizeof(double));
    double* Dld_quartet = (double*) malloc(6 * sizeof(double));
    double* quartet_grads = (double*) malloc(4 * 2 * sizeof(double));
    int* perms = new int[N];
    for (int i = 0; i < N; i++)  perms[i] = i;
    for (int i = 0; i < N * 2; i++) { momentums[i] = 0.; grads[i] = 0.; tmp_params[i] = 0.;}
    


    for (int iteration = 0; iteration < n_iter; iteration++) {
        
        if (iteration > decay_start) {
            temp_ratio = ( (double) (iteration - decay_start)) / (n_iter - decay_start);
            LR = LR_init * (exp(-(temp_ratio * temp_ratio) / decay_cte) + decay_offset); // alternative to classic exponential decay, no big impact in practice
            //LR *= decay;
        }else if (iteration == decay_start) {
             distance_exaggeration = false;
        }

    
        // random quartet designation is done by shuffling perms (which contains np.arange(N) ) 
        std::random_shuffle(&perms[0], &perms[N]);
        
        custom_nestrov_iteration(N, M, Xhd, Xld_flat, tmp_params, grads, momentums, perms, Dhd_quartet, Dld_quartet, quartet_grads, LR, distance_exaggeration);       

    }
    
    free (momentums);
    momentums = NULL;
    free (grads);
    grads = NULL;
    free (tmp_params);
    tmp_params = NULL;
    free (Dhd_quartet);
    Dhd_quartet = NULL;
    free (Dld_quartet);
    Dld_quartet = NULL;
    free (quartet_grads);
    quartet_grads = NULL;
    free (perms);
    perms = NULL;

    return 1;
}








