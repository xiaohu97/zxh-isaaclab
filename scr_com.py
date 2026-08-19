import numpy as np, xml.etree.ElementTree as ET, collections
U='/home/ustczxh/humanoid/zxh-isaaclab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/humanoid_ultra_description/urdf/humanoid_ultra_27dof_description_identified.urdf'
N='/home/ustczxh/humanoid/zxh-isaaclab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand_stand_transition.npz'
r=ET.parse(U).getroot()
ch=collections.defaultdict(list)
for j in r.findall('joint'):
    ch[j.find('parent').get('link')].append(j.find('child').get('link'))
order=[];q=['base_link']
while q:
    n=q.pop(0); order.append(n); q+=ch[n]
idx={n:i for i,n in enumerate(order)}
mass=np.zeros(28); com=np.zeros((28,3))
for l in r.findall('link'):
    n=l.get('name'); i=idx[n]; inr=l.find('inertial')
    if inr is None: continue
    mass[i]=float(inr.find('mass').get('value'))
    o=inr.find('origin')
    com[i]=[float(x) for x in o.get('xyz').split()] if o is not None else [0,0,0]
print('total URDF mass %.4f kg'%mass.sum())
d=np.load(N); P=d['body_pos_w'].astype(np.float64); Q=d['body_quat_w'].astype(np.float64)
def R(q):  # (...,4) wxyz
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],-1),
                     np.stack([2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],-1),
                     np.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)],-2)
Rm=R(Q)                                    # (T,28,3,3)
cw=P+np.einsum('tbij,bj->tbi',Rm,com)      # link CoM world
COM=(cw*mass[None,:,None]).sum(1)/mass.sum()
np.save('/home/ustczxh/humanoid/zxh-isaaclab/_com.npy',COM)
fps=50.0
LA,RA=idx['left_ankle_roll_link'],idx['right_ankle_roll_link']
lz,rz=P[:,LA,2],P[:,RA,2]
print('ankle_roll z: L min %.3f  R min %.3f'%(lz.min(),rz.min()))
gz=min(lz.min(),rz.min())
# contact if ankle_roll frame z within 1.5 cm of its own minimum (foot flat)
tol=0.015
lc=lz<gz+tol; rc=rz<gz+tol
print('frames: L contact %d (%.1f%%)  R contact %d  double %d (%.1f%%)  single-R %d (%.1f%%)  flight %d'%(
   lc.sum(),100*lc.mean(),rc.sum(),(lc&rc).sum(),100*(lc&rc).mean(),(rc&~lc).sum(),100*(rc&~lc).mean(),(~lc&~rc).sum()))
print('CoM height above foot: mean %.4f  min %.4f max %.4f'%((COM[:,2]-gz).mean(),(COM[:,2]-gz).min(),(COM[:,2]-gz).max()))
print('CoM above base_link: mean %.4f'%(COM[:,2]-P[:,0,2]).mean())
print('CoM above ground(0): mean %.4f'%COM[:,2].mean())
# stance width in double support
dw=np.linalg.norm(P[:,LA,:2]-P[:,RA,:2],axis=1)
print('ankle-to-ankle horizontal distance: mean %.4f min %.4f max %.4f'%(dw.mean(),dw.min(),dw.max()))
# CoM horizontal accel
acc=np.gradient(np.gradient(COM,1/fps,axis=0),1/fps,axis=0)
ah=np.linalg.norm(acc[:,:2],axis=1)
print('ref CoM horiz accel |a|: mean %.4f median %.4f p95 %.4f max %.4f m/s2'%(ah.mean(),np.median(ah),np.percentile(ah,95),ah.max()))
ss=rc&~lc
print('  during single-support-R: mean %.4f median %.4f p95 %.4f max %.4f'%(ah[ss].mean(),np.median(ah[ss]),np.percentile(ah[ss],95),ah[ss].max()))
# required CoP offset = CoM_xy - foot; and CoP = COM_xy - h/g * a
h=(COM[:,2]-gz)
cop=COM[:,:2]-(h/9.81)[:,None]*acc[:,:2]
off=cop-P[:,RA,:2]
# express in foot frame (yaw of right ankle)
qq=Q[:,RA]; w,x,y,z=qq[:,0],qq[:,1],qq[:,2],qq[:,3]
yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
c,s=np.cos(-yaw),np.sin(-yaw)
fx=c*off[:,0]-s*off[:,1]; fy=s*off[:,0]+c*off[:,1]
print('CoP offset from R ankle IN FOOT FRAME during single support:')
print('  fore-aft x: mean %.4f  p5 %.4f p95 %.4f  |max| %.4f  (foot half-extent 0.1013)'%(fx[ss].mean(),np.percentile(fx[ss],5),np.percentile(fx[ss],95),np.abs(fx[ss]).max()))
print('  lateral  y: mean %.4f  p5 %.4f p95 %.4f  |max| %.4f  (foot half-extent 0.0370, ankle-roll pos-loop cap 0.0187)'%(fy[ss].mean(),np.percentile(fy[ss],5),np.percentile(fy[ss],95),np.abs(fy[ss]).max()))
print('  frac of single-support frames needing |lateral CoP| > 0.0187: %.3f'%(np.abs(fy[ss])>0.0187).mean())
print('  frac needing |lateral CoP| > 0.0370 (off the foot): %.3f'%(np.abs(fy[ss])>0.0370).mean())
