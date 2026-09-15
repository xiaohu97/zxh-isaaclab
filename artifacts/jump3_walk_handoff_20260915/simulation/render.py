import os
os.environ.setdefault('MUJOCO_GL','osmesa')
from pathlib import Path
import numpy as np
import mujoco, imageio.v2 as imageio
root=Path(__file__).parent
a=np.genfromtxt(root/'matched_b30_nominal.csv',delimiter=',',names=True)
m=mujoco.MjModel.from_xml_path(str(root/'scene_matched.xml'));d=mujoco.MjData(m)
renderer=mujoco.Renderer(m,height=480,width=640)
cam=mujoco.MjvCamera();cam.distance=2.6;cam.azimuth=165;cam.elevation=-12
with imageio.get_writer(root/'jump3_walk_nominal.mp4',fps=30,codec='libx264',quality=8) as writer:
    for i in range(0,len(a),33):
        r=a[i];d.qpos[:]=[r[f'qpos{j}'] for j in range(m.nq)];mujoco.mj_forward(m,d)
        cam.lookat[:]=[d.qpos[0],d.qpos[1],.7]
        renderer.update_scene(d,camera=cam);frame=renderer.render()
        # Draw a minimal time/state caption with Pillow, using replayed physics.
        from PIL import Image,ImageDraw
        im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
        draw.rectangle((8,8,420,37),fill=(0,0,0));draw.text((15,15),f"t={r['time']:.2f}s  {'JUMP3' if r['state']==112 else 'WALK'}  MuJoCo / 0.30s blend",fill='white')
        writer.append_data(np.asarray(im))
renderer.close()
print(root/'jump3_walk_nominal.mp4')
