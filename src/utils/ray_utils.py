from collections import defaultdict
import asyncio
import ray


@ray.remote(max_restarts=-1)
class AsyncLock:
    def __init__(self):
        # one asyncio.Lock per key
        self._locks = defaultdict(asyncio.Lock)

    async def acquire(self, key: str, timeout_sec: int | None = None) -> bool:
        if timeout_sec is None:
            await self._locks[key].acquire()
        else:
            await asyncio.wait_for(self._locks[key].acquire(), timeout=timeout_sec)
        return True

    async def release(self, key: str) -> bool:
        """Release the lock for `key` (must already be acquired)."""
        lock = self._locks[key]
        if lock.locked():
            lock.release()
            return True
        return False
