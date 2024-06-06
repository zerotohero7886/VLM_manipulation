import numpy as np 
import matplotlib.pyplot as plt
from interpolation_tools import d2r, interpolate_and_smooth_nd


""" Set params"""

N,D = 5,3 # number of anchors and dimensions
HZ = 50
dt = 1./HZ 
t_i = .5 # Time increasing rate when the optimization failed.

q_anchors = np.random.uniform(low=d2r(10.0),high=d2r(170.0),size=(N,D))
q_lowers = d2r([-720.]*D)
q_uppers = d2r([180.0]*D)

vel_limit = d2r(60)
acc_limit = d2r(180)
jerk_limit = d2r(360)


""" Interpolate and smooth """

times, traj_interp, traj_smt, times_anchor = interpolate_and_smooth_nd(
    anchor      = q_anchors,
    time_increasing_rate = t_i,
    HZ          = HZ,
    vel_init    = 0.0, ### Not works for other values
    vel_final   = 0.0, ### Not works for other values
    x_lowers    = q_lowers,
    x_uppers    = q_uppers,
    vel_limit   = vel_limit,
    acc_limit   = acc_limit,
    jerk_limit  = jerk_limit,
    verbose     = True,
)


""" Plot and print the result"""

plt.figure(figsize=(6,6))
for d_idx in range(D):
    plt.subplot(D,1,d_idx+1)
    plt.plot([times[0],times[-1]],[q_lowers[d_idx]]*2,
             '--',color='k',lw=1/2,label="lower")
    plt.plot([times[0],times[-1]],[q_uppers[d_idx]]*2,
             '--',color='k',lw=1/2,label="upper")
    plt.plot(times_anchor,q_anchors[:,d_idx],
             'o',mec='k',mfc='none',ms=5,mew=1,label='anchor')
    plt.plot(times,traj_interp[:,d_idx],'--',color='k',lw=1/4,label='interp')
    plt.plot(times,traj_smt[:,d_idx],'-',color='b',lw=1,label='interp+smooth')
    plt.title('dim:[%d]'%(d_idx),fontsize=8)
    plt.ylim([q_lowers[d_idx]-d2r(10),q_uppers[d_idx]+d2r(10)])
    plt.xlabel('Time (sec)',fontsize=8)
    plt.ylabel('Position (rad)',fontsize=8)
    plt.legend(fontsize=8,loc='center left', bbox_to_anchor=(1,0.5))
plt.tight_layout() 
plt.show()


    