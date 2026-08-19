import numpy as np, xml.etree.ElementTree as ET, collections
exec(open('/home/ustczxh/humanoid/zxh-isaaclab/scr_fk.py').read().split('# ---- validate')[0])
d=np.load(N); P=d['body_pos_w'].astype(np.float64); Q=d['body_quat_w'].astype(np.float64)
JP=d['joint_pos'].astype(np.float64); BV=d['body_lin_vel_w'].astype(np.float64)
COM=np.load('/home/ustczxh/humanoid/zxh-isaaclab/_com.npy')
mass=np.zeros(28); comloc=np.zeros((28,3))
for l in ET.parse(U).getroot().findall('link'):
    i=bidx[l.get('name')]; inr=l.find('inertial')
    if inr is None: continue
    mass[i]=float(inr.find('mass').get('value'))
    o=inr.find('origin'); comloc[i]=[float(x) for x in o.get('xyz').split()]
M=mass.sum()
TR=bidx['trunk_link']; RA=bidx['right_ankle_roll_link']; LA=bidx['left_ankle_roll_link']
def quat2R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
rng=np.default_rng(0) if hasattr(np,'default_rng') else np.random.default_rng(0)

def com_of(Pf,Rf):
    return (mass[:,None]*(Pf+np.einsum('bij,bj->bi',Rf,comloc))).sum(0)/M

Nmc=4000
# reset frame distribution: 20% frame0, 40% U(420,494), 40% adaptive(approx uniform over bins)
u=rng.random(Nmc); frames=np.empty(Nmc,int)
m0=u<0.2; m1=(u>=0.2)&(u<0.6); m2=u>=0.6
frames[m0]=0
frames[m1]=rng.integers(420,495,m1.sum())
frames[m2]=((rng.integers(0,19,m2.sum())+rng.random(m2.sum()))/19*940).astype(int)

res={k:[] for k in ['trunk','com','rfoot','lfoot','com_rel_rfoot','rfootz','lfootz','cp_vel','cp_pose','cp_joint','cp_all']}
h=0.936
w0=np.sqrt(9.81/h)
for i in range(Nmc):
    f=frames[i]
    Rb0=quat2R(Q[f,0]); pb0=P[f,0]
    qd0={nm:JP[f,k] for k,nm in enumerate(jorder)}
    Pf0,Rf0=fk(qd0,Rb0,pb0); c0=com_of(Pf0,Rf0)
    # perturbation
    dx=rng.uniform(-0.05,0.05,2); dz=rng.uniform(-0.01,0.01)
    rr,pp,yy=rng.uniform(-0.1,0.1),rng.uniform(-0.1,0.1),rng.uniform(-0.2,0.2)
    dq=rng.uniform(-0.1,0.1,27)
    Rb1=rpy2R(rr,pp,yy)@Rb0
    pb1=pb0+np.array([dx[0],dx[1],dz])
    qd1={nm:JP[f,k]+dq[k] for k,nm in enumerate(jorder)}
    Pf1,Rf1=fk(qd1,Rb1,pb1); c1=com_of(Pf1,Rf1)
    res['trunk'].append(np.linalg.norm(Pf1[TR,:2]-Pf0[TR,:2]))
    res['com'].append(np.linalg.norm(c1[:2]-c0[:2]))
    res['rfoot'].append(np.linalg.norm(Pf1[RA,:2]-Pf0[RA,:2]))
    res['rfootz'].append(Pf1[RA,2]-Pf0[RA,2]); res['lfootz'].append(Pf1[LA,2]-Pf0[LA,2])
    res['com_rel_rfoot'].append(np.linalg.norm((c1[:2]-Pf1[RA,:2])-(c0[:2]-Pf0[RA,:2])))
    # pose-only (no joint noise)
    Pf2,Rf2=fk(qd0,Rb1,pb1); c2=com_of(Pf2,Rf2)
    res['cp_pose'].append(np.linalg.norm((c2[:2]-Pf2[RA,:2])-(c0[:2]-Pf0[RA,:2])))
    # joint-only
    Pf3,Rf3=fk(qd1,Rb0,pb0); c3=com_of(Pf3,Rf3)
    res['cp_joint'].append(np.linalg.norm((c3[:2]-Pf3[RA,:2])-(c0[:2]-Pf0[RA,:2])))
    # velocity: dv_com = v_lin + w x (r_com - r_base)
    vl=rng.uniform(-0.30,0.30,2); wr=np.array([rng.uniform(-0.52,0.52),rng.uniform(-0.52,0.52),rng.uniform(-0.78,0.78)])
    dv=np.array([vl[0],vl[1],rng.uniform(-0.2,0.2)])+np.cross(wr,c0-pb0)
    res['cp_vel'].append(np.linalg.norm(dv[:2])/w0)
for k in ['trunk','com','rfoot','com_rel_rfoot','cp_pose','cp_joint','cp_vel']:
    a=np.array(res[k]); print('%-14s mean %.4f median %.4f p90 %.4f max %.4f'%(k,a.mean(),np.median(a),np.percentile(a,90),a.max()))
for k in ['rfootz','lfootz']:
    a=np.array(res[k]); print('%-14s (signed dz) mean %.4f  p5 %.4f p95 %.4f  |max| %.4f'%(k,a.mean(),np.percentile(a,5),np.percentile(a,95),np.abs(a).max()))
# total capture-point-equivalent disturbance
tot=np.sqrt(np.array(res['com_rel_rfoot'])**2+np.array(res['cp_vel'])**2)
print('capture-point-equivalent TOTAL: mean %.4f median %.4f p90 %.4f'%(tot.mean(),np.median(tot),np.percentile(tot,90)))
print('  velocity-only share of variance: %.3f'%(np.var(res['cp_vel'])/(np.var(res['cp_vel'])+np.var(res['com_rel_rfoot']))))
print('omega0 = %.3f rad/s ; 0.30 m/s -> capture point %.4f m'%(w0,0.30/w0))
