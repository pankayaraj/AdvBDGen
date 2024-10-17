import os

"""
Functions to supress huggiface outputs during trianing
"""
class SuppressOutput:
    def __enter__(self):
        self.stdout = os.dup(1)
        self.stderr = os.dup(2)
        os.close(1)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)
        os.open(os.devnull, os.O_RDWR)

    def __exit__(self, *args):
        os.close(1)
        os.close(2)
        os.dup(self.stdout)
        os.dup(self.stderr)

