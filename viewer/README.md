# OneFlight3D static viewer

The viewer opens real exports from any OneFlight3D run folder. It prefers
`model.glb` / `model.gltf`, falls back to `cloud.ply`, and uses
`confidence_cloud.ply` when present. It includes orbit, pan, zoom, a
confidence overlay toggle, and two-point metric measurements.

The Three.js runtime is vendored in `viewer/vendor`; no CDN is used. Serve this
directory with any static server (do not open `index.html` directly, because
browsers restrict module loading from the filesystem):

```powershell
cd viewer
python -m http.server 8080
```

Open `http://localhost:8080`, then choose a pipeline output folder. The browser
only reads the files selected by the user; no model data is uploaded.
