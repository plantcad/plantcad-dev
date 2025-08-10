"""
Simple child flow for nested flows example.
"""

from metaflow import FlowSpec, step


class ChildFlow(FlowSpec):
    """A simple child flow that processes some data."""

    @step
    def start(self):
        print("Child flow: Processing data...")
        self.result = "processed_data"
        self.next(self.end)

    @step
    def end(self):
        print(f"Child flow complete. Result: {self.result}")


if __name__ == "__main__":
    ChildFlow()
