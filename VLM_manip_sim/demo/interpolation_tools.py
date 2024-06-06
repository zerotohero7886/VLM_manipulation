import cvxpy as cp
import numpy as np


def d2r(degrees):
    """
        Convert Degrees to Radians
    """
    radians = np.deg2rad(degrees)
    return radians


def interpolate_and_smooth_nd(
        anchor, # [N x D]
        initial_time = 1.0,
        time_increasing_rate    = 1., # Increasing time when opt failed.
        HZ                      = 50,
        x_init                  = None,
        x_final                 = None,
        vel_init                = 0.0,
        vel_final               = 0.0,
        x_lowers                = None, # [D]
        x_uppers                = None, # [D]
        vel_limit               = None, # [1]
        acc_limit               = None, # [1]
        jerk_limit              = None, # [1]
        verbose                 = True
    ):
    """
        Interpolate between anchor points
    """
    t_e = initial_time # whole transition time
    dt = 1. / HZ
    N,D = anchor.shape
    while True:
        
        print("\033[33m" + "Try smoothing for {} seconds...".format(t_e) + "\033[0m")

        # linear interpolation
        traj_linear, times, time_anchors, time_anchors_idx = linear_interpolation(anchor, t_e, HZ)
        
        # smoothing 
        traj_smt, opt_fail = smooth_optm_nd(traj_linear, dt=dt, x_init=x_init, x_final=x_final, 
                                            vel_init=vel_init, vel_final=vel_final, x_lowers=x_lowers, 
                                            x_uppers=x_uppers, vel_limit=vel_limit, acc_limit=acc_limit, 
                                            jerk_limit=jerk_limit, idxs_remain=time_anchors_idx, 
                                            vals_remain=anchor, p_norm= 2, verbose=verbose)
        
        if opt_fail:
            t_e += time_increasing_rate
        else:
            print("\033[32m" + "Optimization is successful at {} seconds.".format(t_e) + "\033[0m")
            break
    
    return times, traj_linear, traj_smt, time_anchors  


def linear_interpolation(anchor,t,HZ,verbose=False):
    """ 
        Linear interploation based on max joint velcocity 
    """
    N,D = anchor.shape
    time = np.linspace(0,t,int(t*HZ))

    # Compute max delta
    delta_q = np.zeros((N-1,D))
    for n in range(N-1):
        delta_q[n,:] = anchor[n+1,:] - anchor[n,:]
    delta_q_abs = abs(delta_q)
    delta_q_max = np.max(delta_q_abs,1)
    time_duration = np.zeros(N-1)
    time_idx_len = np.zeros(N-1,dtype=int)

    traj = np.array([])
    for n in range(N-1):
        
        # Divide time section based on max delta in each transition
        time_duration[n] = t * delta_q_max[n]/np.sum(delta_q_max)
        if n == 0:
            time_idx_len[n] = int(time_duration[n]*HZ)
            traj = np.linspace(start=anchor[n],stop=anchor[n+1],num=time_idx_len[n],endpoint=False,axis=0)
        elif n == N-2:
            time_idx_len[n] = len(time) - int(np.sum(time_idx_len[:n]))
            traj_curr = np.linspace(anchor[n],anchor[n+1],time_idx_len[n],endpoint=True,axis=0)
            traj = np.concatenate([traj,traj_curr],axis=0)
        else:
            time_idx_len[n] = int(time_duration[n]*HZ)
            traj_curr = np.linspace(start=anchor[n],stop=anchor[n+1],num=time_idx_len[n],endpoint=False,axis=0)
            traj = np.concatenate([traj,traj_curr],axis=0)
    
    # Concatenate Trajectory
    traj = traj.squeeze()
    
    # Verify anchor
    time_anchors_idx = np.concatenate([[0],np.cumsum(time_idx_len)-1],axis=0)
    time_anchors = time[time_anchors_idx]
    if verbose:
        print("Trajectory Shape: ",traj.shape)
        print("Time anchor: ",time_anchors)
        print("Max velocity for each durations: ",delta_q_max/time_duration)

    return traj, time, time_anchors, time_anchors_idx
    

def get_A_vel_acc_jerk(n,dt):
    A_vel = np.zeros((n,n))
    A_acc = np.zeros((n,n))
    A_jerk = np.zeros((n,n))
    
    # get A_vel
    for i in range(n):
        if i == n-1:
            continue
        else:
            A_vel[i,i+1] = 1/dt
            A_vel[i,i] = -1/dt
    
    # get A_acc
    for i in range(n):
        if i == n-2:
            A_acc[i,i+1] = -1/(dt**2)
            A_acc[i,i] = 1/(dt**2)
        elif i == n-1:
            continue
        else:
            A_acc[i,i+2] = 1/(dt**2)
            A_acc[i,i+1] = -2/(dt**2)
            A_acc[i,i]   = 1/(dt**2)

    # get A_jerk
    for i in range(n):
        if i == n-3:
            A_jerk[i,i+2] = -2/(dt**3)
            A_jerk[i,i+1] = 3/(dt**3)
            A_jerk[i,i] = -1/(dt**3)
        elif i == n-2:
            A_jerk[i,i+1] = 1/(dt**3)
            A_jerk[i,i] = -1/(dt**3)
        elif i == n-1:
            continue
        else:
            A_jerk[i,i+3] = 1/(dt**3)
            A_jerk[i,i+2] = -3/(dt**3)
            A_jerk[i,i+1] = 3/(dt**3)
            A_jerk[i,i] = -1/(dt**3)

    return A_vel, A_acc, A_jerk


def smooth_optm_1d( 
        traj, 
        dt          = 0.1, 
        x_init      = None, 
        x_final     = None, 
        vel_init    = None, 
        vel_final   = None, 
        x_lower     = None, 
        x_upper     = None, 
        vel_limit   = None, 
        acc_limit   = None, 
        jerk_limit  = None, 
        idxs_remain = None, 
        vals_remain = None, 
        p_norm      = 2, 
        verbose     = True, 
    ):
    """ 
        1-D smoothing based on optimization 
    """ 
    n = len(traj) 
    A_pos = np.eye(n,n) 
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    is_optimize_failed = False

    # Objective 
    x = cp.Variable(n) 
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    # objective = cp.Minimize(cp.norm(cp.abs(x-traj)+0.3*cp.abs(A_jerk @ traj),p_norm)

    # Equality constraints 
    A_list,b_list = [],[] 
    if x_init is not None: 
        A_list.append(A_pos[0,:]) 
        b_list.append(x_init) 
    if x_final is not None: 
        A_list.append(A_pos[-1,:])
        b_list.append(x_final) 
    if vel_init is not None: 
        A_list.append(A_vel[0,:]) 
        b_list.append(vel_init) 
    if vel_final is not None: 
        A_list.append(A_vel[-1,:]) 
        b_list.append(vel_final) 
    if idxs_remain is not None: 
        A_list.append(A_pos[idxs_remain,:]) 
        if vals_remain is not None: 
            b_list.append(vals_remain) 
    else: 
        b_list.append(traj[idxs_remain])

    # Inequality constraints
    C_list,d_list = [],[] 
    if x_lower is not None: 
        C_list.append(-A_pos) 
        d_list.append(-x_lower*np.ones(n))
    if x_upper is not None: 
        C_list.append(A_pos) 
        d_list.append(x_upper*np.ones(n))
    if vel_limit is not None: 
        C_list.append(A_vel) 
        C_list.append(-A_vel) 
        d_list.append(vel_limit*np.ones(n)) 
        d_list.append(vel_limit*np.ones(n)) 
    if acc_limit is not None: 
        C_list.append(A_acc) 
        C_list.append(-A_acc) 
        d_list.append(acc_limit*np.ones(n)) 
        d_list.append(acc_limit*np.ones(n)) 
    if jerk_limit is not None: 
        C_list.append(A_jerk) 
        C_list.append(-A_jerk) 
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n)) 
    constraints = []
    if A_list: 
        A = np.vstack(A_list) 
        b = np.hstack(b_list).squeeze() 
        constraints.append(A @ x == b) 
    if C_list: 
        C = np.vstack(C_list) 
        d = np.hstack(d_list).squeeze() 
        constraints.append(C @ x <= d)
    # Solve 
    prob = cp.Problem(objective, constraints) 
    # prob.solve(solver=cp.OSQP,verbose=True)
    prob.solve(solver=cp.CLARABEL,verbose=False)

    # Return 
    traj_smt = x.value

    # Null check 
    if traj_smt is None:
        if verbose: 
            print ("[smooth_optm_1d] Optimization failed.") 
        is_optimize_failed = True
    
    return traj_smt , is_optimize_failed

def smooth_optm_nd(        
        traj, 
        dt          = 0.1, 
        x_init      = None, 
        x_final     = None, 
        vel_init    = None, 
        vel_final   = None, 
        x_lowers    = None, 
        x_uppers    = None, 
        vel_limit   = None, 
        acc_limit   = None, 
        jerk_limit  = None, 
        idxs_remain = None, 
        vals_remain = None, 
        p_norm      = 2, 
        verbose     = True, 
    ):
    L, D = traj.shape
    traj_smt = np.zeros((L,D))
    for d in range(D):
        traj_smt[:,d], is_optimize_failed = smooth_optm_1d( 
            traj        = traj[:,d], 
            dt          = dt, 
            x_init      = vals_remain[0,d],   #x_init, 
            x_final     = vals_remain[-1,d], #x_final,
            vel_init    = vel_init, 
            vel_final   = vel_final, 
            x_lower     = x_lowers[d], 
            x_upper     = x_uppers[d], 
            vel_limit   = vel_limit, 
            acc_limit   = acc_limit, 
            jerk_limit  = jerk_limit, 
            idxs_remain = idxs_remain[1:-1], 
            vals_remain = vals_remain[1:-1,d], 
            p_norm      = p_norm, 
            verbose     = False)
        
        if is_optimize_failed:
            if verbose:
                print("\033[31m" + "Optimization failed at {} seconds.".format(L*dt) + "\033[0m")
            break
        else:
            continue
    return traj_smt,  is_optimize_failed

def quintic_spline_interpolation(t,dt,y1,y2,v2,a2,j2,s2,end=False):
    
    A = np.array([[1,0,0,0,0,0],
                  [1,t,t**2,t**3,t**4,t**5],
                  [0,1,2*t,3*(t**2),4*(t**3),5*(t**4)],
                  [0,0,2,6*t,12*(t**2),20*(t**3)],
                  [0,0,0,6,24*t,60*(t**2)],
                  [0,0,0,0,24,120*t]
                ])        
    b = np.array([y1,y2,v2,a2,j2,s2])
    p = np.linalg.solve(A, b)
    if end :
        times = np.linspace(0,t,int(t/dt),endpoint=True)
    else:
        times = np.linspace(0,t,int(t/dt))
    traj = []
    for i,x in enumerate(times):
        if i == len(times)-1:
          y = p[0] + p[1]*x + p[2]*(x**2) + p[3]*(x**3) + p[4]*(x**4) + p[5]*(x**5)
          v = p[1] + 2*p[2]*x + 3*p[3]*(x**2) + 4*p[4]*(x**3) + 5*p[5]*(x**4)
          a = 2*p[2] + 6*p[3]*x + 12*p[4]*(x**2) + 20*p[5]*(t**3)
          j = 6*p[3] + 24*p[4]*x + 60*p[5]*(t**2)
          s = 24*p[4] + 120*p[5]*(t**2)
          yvajs = [y,v,a,j,s]
          traj.append(y) 
        else:
          y = p[0] + p[1]*x + p[2]*(x**2) + p[3]*(x**3) + p[4]*(x**4) + p[5]*(x**5)
          traj.append(y)
    return traj, yvajs
    

