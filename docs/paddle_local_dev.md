# Paddle Local Development (No Docker)

This guide is for running PaddleDetection training workers in local development.

## 1) Install dependencies

Use a clean Python virtual environment first.

```bash
pip install -r requirements/worker.common.txt
pip install -r requirements/worker.paddle.local.txt
```

Install one Paddle runtime:

```bash
# CPU runtime
pip install paddlepaddle==2.6.2

# OR GPU runtime (CUDA)
pip install paddlepaddle-gpu==2.6.2
```

Install PaddleDetection runtime:

```bash
pip install --no-deps paddledet==2.6.0
pip install pycocotools typeguard "visualdl>=2.2.0" lap terminaltables Cython
pip install "protobuf<=3.20.3"
```

## 2) Prepare PaddleDetection repo

Clone PaddleDetection directly under backend root (same level as `train_platform/`):

```bash
git clone --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git ./PaddleDetection
```

No runtime env-var export is required. The trainer uses this default local path.

## 3) Run API + Paddle worker

```bash
python -m alembic -c alembic.ini upgrade head
python -m uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload
```

In another terminal:

```bash
python -m train_platform.workers.paddle_worker
```

## 4) Quick verification

After creating a training run with a Paddle architecture:

- Worker log should print `engine=paddle-det` claim/start info.
- Run directory should contain `coco/train.json` and `coco/val.json`.
- Final artifacts should include `weights/best.pdparams` and `weights/last.pdparams`.
