class DummyTrainer:
    def train(self, ctx):
        for epoch in range(ctx.epochs):
            ctx.report_metrics(epoch, {"loss": float(ctx.epochs - epoch) / max(ctx.epochs, 1)})
            if ctx.should_cancel():
                return
