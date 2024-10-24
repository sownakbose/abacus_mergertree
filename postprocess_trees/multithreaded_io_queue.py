"""
A simple multi-threaded Python file writer.

Author: Lehman Garrison (lgarrison.github.io)
"""

import threading
import queue

import numpy as np


class MultithreadedIOQueue:
    """
    Multi-threaded file writer.  Writes in Numpy `npy` format
    by default.

    Usage
    -----
    >>> from multithreaded_io_queue import MultithreadedIOQueue
    >>> write_queue = MultithreadedIOQueue(4)
    >>> write_queue.write(fn, data)
    >>> ... # more writes
    >>> write_queue.close()
    """

    def __init__(self, nthreads):
        self._queue = queue.SimpleQueue()
        self._threads = [
            threading.Thread(target=self._threadloop) for i in range(nthreads)
        ]
        for t in self._threads:
            t.start()
        self._closed = False

    def __del__(self):
        self.close()

    def _threadloop(self):
        while (data := self._queue.get()) is not None:
            fn = data["fn"]
            arr = data["array"]
            np.save(fn, arr)

    def write(self, fn, array):
        assert not self._closed
        self._queue.put(dict(fn=fn, array=array))

    def close(self):
        for i in range(len(self._threads)):
            self._queue.put(None)  # poison
        for t in self._threads:
            t.join()
        self._closed = True
