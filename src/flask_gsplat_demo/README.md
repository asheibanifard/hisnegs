# Flask gsplat Demo

Interactive demo for rotating Gaussian splats from checkpoint:

- Default checkpoint: `src/checkpoints/render_skpt/splat_step10000.pt`
- UI endpoint: `http://<server>:8000`

## Local run (existing environment)

```bash
cd src/flask_gsplat_demo
export SPLAT_CKPT=../checkpoints/render_skpt/splat_step10000.pt
python app.py
```

## Docker run

```bash
cd src/flask_gsplat_demo
docker compose up --build
```

For remote server demo, open port `8000` on your firewall/security group.

## Notes

- First request may be slower because `gsplat` compiles its CUDA extension JIT.
- The app supports controls for yaw, pitch, radius, fov, and max gaussians.
