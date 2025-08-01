#!/usr/bin/env python3

"""
Metaflow Example: Simple bedtools sorting with Docker

Demonstrates using biocontainers/bedtools via Docker SDK to sort genomic intervals.
"""

import tempfile
import os
import docker
from metaflow import FlowSpec, step

# Example unsorted BED data to be sorted
BED_DATA = """chr2	4000	5000	region6	180	+
chr1	3000	4000	region3	150	-
chr2	500	1500	region4	300	+
chr1	1000	2000	region1	100	+
chr2	1200	2200	region5	250	-
chr1	1500	2500	region2	200	+"""


class BedtoolsFlow(FlowSpec):
    """
    Simple Metaflow flow that sorts BED intervals using bedtools in Docker.

    Shows: Docker SDK integration, file mounting, and capturing container output.
    """

    @step
    def start(self):
        """Create sample genomic data and start the flow."""
        print("Creating sample BED data for sorting...")

        # Sample genomic intervals (unsorted)
        self.bed_data = BED_DATA

        n_intervals = len(self.bed_data.strip().split("\n"))
        print(f"Created {n_intervals} genomic intervals")
        self.next(self.sort_intervals)

    @step
    def sort_intervals(self):
        """Sort BED intervals using bedtools in Docker container."""
        print("Sorting genomic intervals with bedtools...")

        # Initialize Docker client and create temporary file
        client = docker.from_env()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bed", delete=False
        ) as tmp_file:
            tmp_file.write(self.bed_data)
            tmp_bed_file = tmp_file.name

        try:
            # Mount host directory into container at /data
            # This allows bedtools inside container to access our temp file
            host_dir = os.path.dirname(tmp_bed_file)
            filename = os.path.basename(tmp_bed_file)

            print(f"Mounting {host_dir} -> /data in container")

            # Use bedtools sort to sort genomic intervals
            sort_result = client.containers.run(
                "biocontainers/bedtools:v2.27.1dfsg-4-deb_cv1",
                command=["bedtools", "sort", "-i", f"/data/{filename}"],
                volumes={host_dir: {"bind": "/data", "mode": "ro"}},  # Read-only mount
                remove=True,  # Auto-cleanup container after execution
            )

            self.sorted_intervals = sort_result.decode().strip()

        finally:
            os.unlink(tmp_bed_file)  # Cleanup temp file

        self.next(self.end)

    @step
    def end(self):
        """Display sorted results."""
        original_count = len(self.bed_data.strip().split("\n"))

        print("\n=== Bedtools Sort Results ===")
        print(f"Original intervals: {original_count}")
        print("\nSorted intervals:")
        print(self.sorted_intervals)

        print("\n✓ Successfully used biocontainers/bedtools via Docker SDK")
        print("✓ Demonstrated file mounting: host -> container")
        print("✓ Captured and processed container output")


if __name__ == "__main__":
    BedtoolsFlow()
