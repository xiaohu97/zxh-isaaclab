from pathlib import Path
import xml.etree.ElementTree as ET
import json
BASE=Path('/home/ustczxh/humanoid/unitree_mujoco/unitree_robots/g1')
OUT=Path(__file__).parent
URDF=Path('/home/ustczxh/humanoid/zxh-isaaclab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid/g1_29dof_rev_1_0_identified0914_ball.urdf')
src=ET.parse(URDF).find("./link[@name='left_rubber_hand']/inertial")
assert src.find('origin').get('rpy','0 0 0')=='0 0 0'
for matched in (False, True):
    tree=ET.parse(BASE/'g1_29dof_identified0907.xml')
    root=tree.getroot()
    inertial=root.find(".//body[@name='left_rubber_hand']/inertial")
    inertial.attrib.clear()
    inertia=src.find('inertia')
    inertial.attrib.update(mass=src.find('mass').get('value'),pos=src.find('origin').get('xyz'),fullinertia=' '.join(inertia.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')))
    if matched:
        limits=[88,139,88,139,25,25,88,139,88,139,25,25,88,25,25,25,25,25,25,25,5,5,25,25,25,25,25,5,5]
        for motor,lim in zip(root.findall('./actuator/motor'),limits):
            motor.set('ctrlrange',f'{-lim} {lim}')
            joint=root.find(f".//worldbody//joint[@name='{motor.get('joint')}']")
            joint.set('actuatorfrcrange',f'{-lim} {lim}')
            joint.set('damping','0')
            joint.set('frictionloss','0')
            joint.set('armature','0.01')
    name='matched' if matched else 'sdk'
    path=OUT/f'robot_{name}.xml'
    tree.write(path)
    scene=ET.parse(BASE/'scene_identified0907.xml')
    scene.find('./include').set('file',str(path))
    scene.write(OUT/f'scene_{name}.xml')
print('Wrote private SDK / training-effort-and-passive-dynamics variants with 0914 hand inertia')
