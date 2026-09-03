# Local frozen inputs

Clinical graph caches and binary masks are excluded from the public package.
Set `SEER_PRIVATE_ARTIFACT_ROOT` to an authorized external artifact tree.
Reference-checkpoint evaluation fails if these required inputs are absent; it
does not silently substitute different graph inputs. See `docs/security.md`.
