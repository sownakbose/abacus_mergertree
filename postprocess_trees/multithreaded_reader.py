import numpy as np
import threading
import queue


def multithreaded_read(fns, nthreads=2):
    """
    A simple multithreaded reader that spins up N threads
    to read a list of files, returning the result in a list.
    Reads in Numpy format by default.

    This doesn't allow for asynchronous IO (i.e. this function is
    blocking), but this keeps the code dead simple.

    Parameters
    ----------
    fns: list of str
        List of files to read
    nthreads: int, optional
        Number of threads to use

    Returns
    -------
    data: list of object
        List of return values from np.load, same order as `fns`
    """
    fn_queue = queue.SimpleQueue()
    data_queue = queue.SimpleQueue()

    def _threadloop():
        while (fn := fn_queue.get()) is not None:
            data_queue.put((fn, np.load(fn)))

    threads = [threading.Thread(target=_threadloop) for i in range(nthreads)]
    for t in threads:
        t.start()
    for fn in fns:
        fn_queue.put(fn)
    for t in threads:
        fn_queue.put(None)
    for t in threads:
        t.join()
    res = dict([data_queue.get() for i in range(len(fns))])
    assert fn_queue.empty()
    assert data_queue.empty()
    res = [res[fn] for fn in fns]
    return res  # list of data, same order as input
