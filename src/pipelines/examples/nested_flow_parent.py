"""
Simple example of a Metaflow Flow that runs another Flow using Runner.
"""

from metaflow import FlowSpec, step, Runner


class ParentFlow(FlowSpec):
    """A parent flow that runs the ChildFlow using Runner."""

    @step
    def start(self):
        print("Parent flow: Starting...")
        self.next(self.run_child_flow)

    @step
    def run_child_flow(self):
        print("Parent flow: Running child flow...")

        # Use Runner to execute the child flow
        child_flow_file = __file__.replace(
            "nested_flow_parent.py", "nested_flow_child.py"
        )
        with Runner(child_flow_file).run() as running:
            # Access the run object and get data from the end step
            run = running.run
            self.child_result = run["end"].task.data.result

        print(f"Parent flow: Got result from child: {self.child_result}")
        self.next(self.end)

    @step
    def end(self):
        print("Parent flow: Complete!")


if __name__ == "__main__":
    ParentFlow()
