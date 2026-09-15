import subprocess, json
from pathlib import Path
root=Path(__file__).parent
cases=[]
for plant in ['matched','sdk']:
    for blend in [.30,.10]:
        for tag,dv,dw in [('nominal',0,0),('forward',.25,.75),('backward',-.25,-.75)]:
            cases.append((plant,blend,tag,dv,dw))
for plant,blend,tag,dv,dw in cases:
    name=f'{plant}_b{round(blend*100):02d}_{tag}'
    with (root/f'{name}.log').open('w') as log:
        result=subprocess.run([str(root/'build/sim'),str(root/f'scene_{plant}.xml'),str(root/f'{name}.csv'),'jump',str(blend),'6',str(dv),str(dw)],stdout=log,stderr=subprocess.STDOUT)
    lines=(root/f'{name}.log').read_text().splitlines()
    print(name,'exit',result.returncode,lines[-1],flush=True)
