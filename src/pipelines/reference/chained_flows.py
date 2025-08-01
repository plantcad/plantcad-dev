from metaflow import FlowSpec, step, trigger_on_finish


@trigger_on_finish(flow="FirstFlow")
class SecondFlow(FlowSpec):
    @step
    def start(self):
        print("This is the second flow")
        self.next(self.end)

    @step
    def end(self):
        pass
