import meshlib.mrmeshpy as mrmeshpy
import os
import sys

path = "./meshes"#获取制定路径
dirs = os.listdir(path)
# Load mesh


def simpilfy(file):

    mesh = mrmeshpy.loadMesh( path + "/" + file)

# Repack mesh optimally.
# It's not necessary but highly recommended to achieve the best performance in parallel processing
    mesh.packOptimally()

# Setup decimate parameters
    settings = mrmeshpy.DecimateSettings()

# Decimation stop thresholds, you may specify one or both
# settings.maxDeletedFaces = 1000 # Number of faces to be deleted
    settings.maxError = 0.0002 # Maximum error when decimation stops

# Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
# Recommended to set to number of CPU cores or more available for the best performance
    settings.subdivideParts = 10

# Simplify mesh
    mrmeshpy.decimateMesh(mesh, settings)

# Save result
    mrmeshpy.saveMesh(mesh, "./简化meshes" + "/" + file)




for i in dirs:
    if os.path.splitext(i)[1] == ".STL":
        if os.path.getsize(path + "/" + i) > 100 *1024:#大于100KB
            simpilfy(i)
        else:
            pass
            

sys.exit()





