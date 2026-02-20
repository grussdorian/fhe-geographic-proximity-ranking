import unittest
import os
import sys
import platform
import datetime
import socket

class TestLogger(unittest.TextTestRunner):
    """
    A custom TextTestRunner that logs failures and errors to a file
    in the 'logs' folder, but only if the --log flag is present.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_enabled = "--log" in sys.argv
        self.log_dir = "logs"
        self.report_file = None

    def run(self, test):
        # Run tests as usual
        result = super().run(test)

        # If logging is enabled and there are failures or errors, generate a report
        if self.log_enabled and (result.failures or result.errors):
            self._generate_report(result)
        
        return result

    def _generate_report(self, result):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp}.log"
        self.report_file = os.path.join(self.log_dir, filename)

        with open(self.report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("FHE PROXIMITY TEST FAILURE REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Device: {socket.gethostname()}\n")
            f.write(f"OS: {platform.system()} {platform.release()}\n")
            f.write(f"Arch: {platform.machine()}\n")
            f.write(f"Python: {platform.python_version()}\n")
            f.write(f"Processor: {platform.processor()}\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tests Run: {result.testsRun}\n")
            f.write(f"Failures: {len(result.failures)}\n")
            f.write(f"Errors: {len(result.errors)}\n\n")

            if result.errors:
                f.write("ERRORS\n")
                f.write("-" * 20 + "\n")
                for test, err in result.errors:
                    f.write(f"Test: {test}\n")
                    f.write(f"Traceback:\n{err}\n")
                    f.write("-" * 40 + "\n")
                f.write("\n")

            if result.failures:
                f.write("FAILURES\n")
                f.write("-" * 20 + "\n")
                for test, fail in result.failures:
                    f.write(f"Test: {test}\n")
                    f.write(f"Details:\n{fail}\n")
                    f.write("-" * 40 + "\n")
                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")

        print(f"\n[INFO] Failure/Error report generated: {self.report_file}")

def get_runner():
    """
    Returns the appropriate runner based on command line flags.
    If --log is present, returns TestLogger. Otherwise, returns TextTestRunner.
    """
    if "--log" in sys.argv:
        # Remove --log from sys.argv so unittest.main() doesn't complain if it's used elsewhere,
        # though we are using manual suite running in these files.
        return TestLogger(verbosity=2)
    return unittest.TextTestRunner(verbosity=2)
