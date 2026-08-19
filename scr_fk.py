import numpy as np, xml.etree.ElementTree as ET, collections
U='/home/ustczxh/humanoid/zxh-isaaclab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/humanoid_ultra_description/urdf/humanoid_ultra_27dof_description_identified.urdf'
N='/home/ustczxh/humanoid/zxh-isaaclab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand_stand_transition.npz'
root=ET.parse(U).getroot()
ch=collections.defaultdict(list); jinfo={}
for j in root.findall('joint'):
    p=j.find('parent').get('link'); c=j.find('child').get('link')
    o=j.find('origin'); xyz=np.array([float(v) for v in o.get('xyz').split()]) if o is not None else np.zeros(3)
    rpy=np.array([float(v) for v in o.get('rpy').split()]) if (o is not None and o.get('rpy')) else np.zeros(3)
    ax=j.find('axis'); axis=np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.array([0.,0.,1.])
    ch[p].append(c); jinfo[c]=(j.get('name'),p,xyz,rpy,axis,j.get('type'))
order=[]; q=['base_link']
while q:
    n=q.pop(0); order.append(n); q+=ch[n]
bidx={n:i for i,n in enumerate(order)}
jorder=[jinfo[n][0] for n in order[1:]]   # joint order in BFS-link order
def rpy2R(r,p,y):
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])
def axang(a,t):
    a=a/np.linalg.norm(a); K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
def fk(qd, Rb=np.eye(3), pb=np.zeros(3)):
    """qd: dict jointname->angle. returns (28,3) pos, (28,3,3) rot in world"""
    P=np.zeros((28,3)); R=np.zeros((28,3,3)); P[0]=pb; R[0]=Rb
    for i,n in enumerate(order):
        if i==0: continue
        jn,par,xyz,rpy,axis,typ=jinfo[n]; pi=bidx[par]
        Rj=R[pi]@rpy2R(*rpy)
        Pj=P[pi]+R[pi]@xyz
        th=qd.get(jn,0.0) if typ!='fixed' else 0.0
        R[i]=Rj@axang(axis,th); P[i]=Pj
    return P,R
# ---- validate against npz
d=np.load(N); P=d['body_pos_w'].astype(np.float64); Q=d['body_quat_w'].astype(np.float64); JP=d['joint_pos'].astype(np.float64)
# need the joint-name order used in joint_pos. Try isaaclab-ish: assume BFS joint order (jorder)
def quat2R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
t=500
for name,names in [('BFS',jorder)]:
    qd={nm:JP[t,k] for k,nm in enumerate(names)}
    Pf,_=fk(qd,quat2R(Q[t,0]),P[t,0])
    print(name,'max FK-vs-npz pos err %.5f m'%np.abs(Pf-P[t]).max())
print('joint order used:',jorder)
np.save('/home/ustczxh/humanoid/zxh-isaaclab/_jorder.npy',np.array(jorder))
