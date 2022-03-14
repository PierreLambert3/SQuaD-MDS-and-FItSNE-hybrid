import numpy as np
import numba
from sklearn.decomposition import PCA

def populate_hparams(hparams, N):
    if not 'on N PC' in hparams:
        hparams['on N PC'] = -1

    if not 'n iter' in hparams:
        hparams['n iter'] = 3000 # higher values bring better results

    if not 'LR' in hparams:
        hparams['LR'] = 1

def run_SQuaD_MDS(Xhd, hparams, python=False):
    if not python:
        populate_hparams(hparams, Xhd.shape[0])
        from FItSNE_files.fast_tsne import fast_tsne
        return fast_tsne(Xhd, method_type="SQuaD_MDS", perplexity_list = [1], max_iter = hparams['n iter'])
    else:
        return run_SQuaD_MDS_python(Xhd, hparams)

def run_SQuaD_MDS_python(Xhd, hparams, progress_stuff=None):
    N, M = Xhd.shape
    populate_hparams(hparams, N) # set the missing hyperparameters with their default values

    if int(hparams['on N PC']) < Xhd.shape[1] and int(hparams['on N PC']) > 1: #if the HD data has A LOT of dimensions, using their principal components can speed up the optimisation for a negligible cost given a number of PC sufficiently high and an edaquate intrinsic dimensionality
        Xhd = PCA(n_components=int(hparams['on N PC']), whiten=True, copy=True).fit_transform(Xhd).astype(np.float64)

    Xld = init_embedding(Xhd) # init Xld with PCA, then set standard dev of the initialisation to 10

    # hparams['LR'] = max(2., 0.005*N)
    print("LR : ", hparams['LR'])

    LR_init = hparams['LR']
    LR      = LR_init
    N_iter  = hparams['n iter']
    decay_start = int(0. * N_iter)  # if random init: it is recommended to start the decay later to give the time for the global config to adjust with big steps
    distance_exaggeration = False
    if decay_start > 0:
        distance_exaggeration = True # use squared distances in the HD quartet: can be usefull to start a couple of iterations like this if random init

    decay_cte = 0.34
    offset = -np.exp(-1/decay_cte)

    momentums     = np.zeros((N, 2))
    grads         = np.zeros((N, 2), dtype=np.float32)
    perms         = np.arange(N)
    batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    Dhd_quartet   = np.zeros((6,))

    for i in range(N_iter):
        if i > decay_start:
            ratio = (i - decay_start) / (N_iter - decay_start)
            LR = LR_init * (np.exp(-(ratio*ratio) / decay_cte) + offset)

        elif i == decay_start:
            distance_exaggeration = False
        np.random.shuffle(perms) # used for the random quartet designation
        nestrov_iteration(Xld, grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR, distance_exaggeration=distance_exaggeration)

    return Xld


def nestrov_iteration(Xld, grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR, distance_exaggeration=False): # coputes and applies gradients, updates momentum too
    momentums *= 0.99

    fill_MDS_grads(Xld+momentums, grads, perms, batches_idxes, Xhd, Dhd_quartet, exaggeration=distance_exaggeration)

    # norm = np.linalg.norm(grads, keepdims=True)
    # mul  = LR / norm
    mul = LR

    momentums -= mul * grads
    Xld += momentums


def run_SQuaD_MDS_version2(Xhd, hparams, progress_stuff=None):
    N, M = Xhd.shape
    populate_hparams(hparams, N) # set the missing hyperparameters with their default values

    if int(hparams['on N PC']) < Xhd.shape[1] and int(hparams['on N PC']) > 1: #if the HD data has A LOT of dimensions, using their principal components can speed up the optimisation for a negligible cost given a number of PC sufficiently high and an edaquate intrinsic dimensionality
        Xhd = PCA(n_components=int(hparams['on N PC']), whiten=True, copy=True).fit_transform(Xhd).astype(np.float64)



    Xld = init_embedding(Xhd) # init Xld with PCA, then set standard dev of the initialisation to 10

    hparams['LR'] = max(2., 0.005*N)
    print("LR : ", hparams['LR'])

    LR_init = hparams['LR']
    LR      = LR_init
    N_iter  = hparams['n iter']


    decay_cte = 0.34
    offset = -np.exp(-1/decay_cte)

    momentums     = np.zeros((N, 2))
    target_positions = Xld.copy()
    grads         = np.zeros((N, 2), dtype=np.float32)
    perms         = np.arange(N)
    batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    Dhd_quartet   = np.zeros((6,))


    if progress_stuff is None:
        for i in range(N_iter):
            ratio = i / N_iter
            mul = (np.exp(-(ratio*ratio) / decay_cte) + offset)
            LR = LR_init * mul

            np.random.shuffle(perms) # used for the random quartet designation
            nestrov_iteration_version2(mul, i/N_iter, target_positions, Xld, grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR)
    else:
        progress_listener, instance = progress_stuff
        for i in range(N_iter):
            ratio = i / N_iter
            mul = (np.exp(-(ratio*ratio) / decay_cte) + offset)
            LR = LR_init * mul

            np.random.shuffle(perms) # used for the random quartet designation
            nestrov_iteration_version2(mul, i/N_iter, target_positions, Xld, grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR)
            if i % 10 == 0:
                progress_listener.notify((instance.dataset_name, instance.proj_name, Xld, instance), [])
    return Xld


def nestrov_iteration_version2(decaying_value, iteration_ratio, target_positions, Xld, grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR, distance_exaggeration=False): # coputes and applies gradients, updates momentum too
    LR = 10000
    base_alpha = 0.996
    alpha = base_alpha + ((1-base_alpha) * (1 - decaying_value))
    fill_MDS_grads(Xld, grads, perms, batches_idxes, Xhd, Dhd_quartet, exaggeration=distance_exaggeration)
    target_positions *= alpha
    target_positions += (1 - alpha) * (Xld - LR*grads)
    Xld += 0.5 * (target_positions - Xld)



def init_embedding(Xhd, target = 10.0):
    Xld  = PCA(n_components=2, whiten=True, copy=True).fit_transform(Xhd).astype(np.float64)
    Xld *= target/np.std(Xld)
    return Xld

@numba.jit(nopython=True, fastmath=True)
def fill_MDS_grads(X_LD, grad_acc, perms, batches_idxes, Xhd, Dhd_quartet, exaggeration=False, zero_grad=True):
    if zero_grad:
        grad_acc.fill(0.)
    for batch_idx in batches_idxes:
        quartet = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if exaggeration:
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = np.sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))
        Dhd_quartet   /= np.sum(Dhd_quartet) + 6*1e-12


        quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)

        grad_acc[quartet[0], 0] += quartet_grads[0]
        grad_acc[quartet[0], 1] += quartet_grads[1]
        grad_acc[quartet[1], 0] += quartet_grads[2]
        grad_acc[quartet[1], 1] += quartet_grads[3]
        grad_acc[quartet[2], 0] += quartet_grads[4]
        grad_acc[quartet[2], 1] += quartet_grads[5]
        grad_acc[quartet[3], 0] += quartet_grads[6]
        grad_acc[quartet[3], 1] += quartet_grads[7]


# quartet gradients for a 2D projection, Dhd contains the top-right triangle of the HD distances
# the points are named a,b,c and d internaly to keep track of who is who
# points shape: (4, 2)
# Dhd shape   : (6,)
@numba.jit(nopython=True)
def compute_quartet_grads(points, Dhd):
    xa, ya = points[0]
    xb, yb = points[1]
    xc, yc = points[2]
    xd, yd = points[3]

    # LD distances, add a small number just in case
    d_ab = np.sqrt((xa-xb)**2 + (ya-yb)**2) + 1e-12
    d_ac = np.sqrt((xa-xc)**2 + (ya-yc)**2) + 1e-12
    d_ad = np.sqrt((xa-xd)**2 + (ya-yd)**2) + 1e-12
    d_bc = np.sqrt((xb-xc)**2 + (yb-yc)**2) + 1e-12
    d_bd = np.sqrt((xb-xd)**2 + (yb-yd)**2) + 1e-12
    d_cd = np.sqrt((xc-xd)**2 + (yc-yd)**2) + 1e-12

    # HD distances
    pab, pac, pad, pbc, pbd, pcd = Dhd[0], Dhd[1], Dhd[2], Dhd[3], Dhd[4], Dhd[5]

    # for each element of the sum: use the same gradient function and just permute the points given in input
    gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD = ABCD_grad(
                                                    xa, ya, xb, yb, xc, yc, xd, yd,\
                                                    d_ab, d_ac, d_ad, d_bc, d_bd, d_cd,\
                                                    pab)


    gxA2, gyA2, gxC2, gyC2, gxB2, gyB2, gxD2, gyD2 = ABCD_grad(
                                                    xa, ya, xc, yc, xb, yb, xd, yd,\
                                                    d_ac, d_ab, d_ad, d_bc, d_cd, d_bd,\
                                                    pac)


    gxA3, gyA3, gxD3, gyD3, gxC3, gyC3, gxB3, gyB3 = ABCD_grad(
                                                    xa, ya, xd, yd, xc, yc, xb, yb,\
                                                    d_ad, d_ac, d_ab, d_cd, d_bd, d_bc,\
                                                    pad)


    gxB4, gyB4, gxC4, gyC4, gxA4, gyA4, gxD4, gyD4 = ABCD_grad(
                                                    xb, yb, xc, yc, xa, ya, xd, yd,\
                                                    d_bc, d_ab, d_bd, d_ac, d_cd, d_ad,\
                                                    pbc)


    gxB5, gyB5, gxD5, gyD5, gxA5, gyA5, gxC5, gyC5 = ABCD_grad(
                                                    xb, yb, xd, yd, xa, ya, xc, yc,\
                                                    d_bd, d_ab, d_bc, d_ad, d_cd, d_ac,\
                                                    pbd)


    gxC6, gyC6, gxD6, gyD6, gxA6, gyA6, gxB6, gyB6 = ABCD_grad(
                                                    xc, yc, xd, yd, xa, ya, xb, yb,\
                                                    d_cd, d_ac, d_bc, d_ad, d_bd, d_ab,\
                                                    pcd)

    gxA = gxA + gxA2 + gxA3 + gxA4 + gxA5 + gxA6
    gyA = gyA + gyA2 + gyA3 + gyA4 + gyA5 + gyA6

    gxB = gxB + gxB2 + gxB3 + gxB4 + gxB5 + gxB6
    gyB = gyB + gyB2 + gyB3 + gyB4 + gyB5 + gyB6

    gxC = gxC + gxC2 + gxC3 + gxC4 + gxC5 + gxC6
    gyC = gyC + gyC2 + gyC3 + gyC4 + gyC5 + gyC6

    gxD = gxD + gxD2 + gxD3 + gxD4 + gxD5 + gxD6
    gyD = gyD + gyD2 + gyD3 + gyD4 + gyD5 + gyD6

    return gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD


# gradients for one element of the loss function's sum
@numba.jit(nopython=True)
def ABCD_grad(xa, ya, xb, yb, xc, yc, xd, yd, dab, dac, dad, dbc, dbd, dcd, pab):
    sum_dist    = dab + dac + dad + dbc + dbd + dcd

    ratio1 = dab/sum_dist
    twice_ratio = 2*((pab - ratio1)/sum_dist)

    gxA = twice_ratio * (ratio1 * ((xa-xb)/dab + (xa-xc)/dac + (xa-xd)/dad ) - (xa-xb)/dab )
    gyA = twice_ratio * (ratio1 * ((ya-yb)/dab + (ya-yc)/dac + (ya-yd)/dad ) - (ya-yb)/dab )

    gxB = twice_ratio * (ratio1 * ((xb-xa)/dab + (xb-xc)/dbc + (xb-xd)/dbd ) - (xb-xa)/dab )
    gyB = twice_ratio * (ratio1 * ((yb-ya)/dab + (yb-yc)/dbc + (yb-yd)/dbd ) - (yb-ya)/dab )

    gxC = twice_ratio * (ratio1 * ((xc-xa)/dac + (xc-xb)/dbc + (xc-xd)/dcd ))
    gyC = twice_ratio * (ratio1 * ((yc-ya)/dac + (yc-yb)/dbc + (yc-yd)/dcd ))

    gxD = twice_ratio * (ratio1 * ((xd-xa)/dad + (xd-xb)/dbd + (xd-xc)/dcd ))
    gyD = twice_ratio * (ratio1 * ((yd-ya)/dad + (yd-yb)/dbd + (yd-yc)/dcd ))

    return gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD
