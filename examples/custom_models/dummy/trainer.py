from pathlib import Path


class DummyTrainer:
    def train(self, ctx):
        print("ordinary user output", flush=True)
        weights_dir = Path(ctx.output_dir) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_path = weights_dir / "best_dummy.pth"
        last_path = weights_dir / "latest_dummy.ckpt"
        best_path.write_text("dummy best weights\n", encoding="utf-8")
        last_path.write_text("dummy latest weights\n", encoding="utf-8")
        ctx.report_artifact(role="best_weights", path="weights/best_dummy.pth", format="pytorch")
        ctx.report_artifact(role="last_weights", path="weights/latest_dummy.ckpt", format="pytorch")
        for epoch in range(ctx.epochs):
            ctx.report_metrics(epoch, {"loss": float(ctx.epochs - epoch) / max(ctx.epochs, 1)})
            if ctx.should_cancel():
                return
