from __future__ import annotations
from threading import Lock
from typing import Dict

class LazyLock:
    """Implementation of per-element locking for large arrays.
    Locks are allocated on-demand when their key is first called.
    Once created, a lock is never destroyed. to clear it from memory, the LazyLock object must be destroyed."""
    def __init__(self):
        self._dict_lock = Lock()                # global lock on the _locks dict
        self._locks: Dict[int, Lock] = dict()   # key is the id of the lock and its value is 'True' if it's locked
                                                # or 'False' otherwise

    def _ensure(self, key) -> None:
        """Ensure the actual lock exists. If not, create it."""
        if self._locks.get(key) is None:            # if the lock doesn't exist
            with self._dict_lock:                   # lock dict
                if self._locks.get(key) is None:    # If lock hasn't been created while waiting for _dict_lock
                    self._locks[key] = Lock()       # create lock


    def __getitem__(self, key: int) -> Lock:
        """
        Get a lock for the given key.
        Locks are created on-demand, if the lock for the given key does not exist, it is created.
        :param key: key index, must be int
        :return: threading.Lock
        """
        if type(key) != int:
            raise KeyError("Given key is not int")
        self._ensure(key)
        return self._locks[key]


