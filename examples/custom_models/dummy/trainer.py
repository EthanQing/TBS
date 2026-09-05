class DummyTrainer:
    def train(self, ctx):
        print("ordinary user output", flush=True)
        for epoch in range(ctx.epochs):
            ctx.report_metrics(epoch, {"loss": float(ctx.epochs - epoch) / max(ctx.epochs, 1)})
            if ctx.should_cancel():
                return
