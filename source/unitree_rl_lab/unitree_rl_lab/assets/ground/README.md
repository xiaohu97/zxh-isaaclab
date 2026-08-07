# Isaac Sim grid ground

This directory contains the standard Isaac Sim 5.0 grid ground used by
`GroundPlaneCfg`.  It is kept locally so environment creation does not depend
on the runtime availability of the NVIDIA asset server.

Original asset URL:

`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd`

The relative `Materials/Textures` layout must be preserved because the USD
references these textures directly.
