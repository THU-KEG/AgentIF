
import pickle
import os
import time
import threading
from filelock import FileLock

class Cache(object):
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.lock = threading.Lock()  # 添加线程锁

    def check_prompt(self, prompt):
        prompt = prompt.strip()
        cache_key = prompt
        Flag = False
        with self.lock:
            if cache_key in self.cache_dict:
                Flag = True
                if self.cache_dict[cache_key] != None:
                    return self.cache_dict[cache_key]

        return None
    
    def save_prompt(self, prompt, response):
        prompt = prompt.strip()
        cache_key = prompt
        with self.lock:
            self.cache_dict[cache_key] = response
            self.add_n += 1
        
        return

    def save_cache(self):
        if self.add_n == 0:
            return

        with self.lock:
            latest_cache = self.load_cache(allow_retry=False)
            for k, v in latest_cache.items():
                self.cache_dict[k] = v

            lock = FileLock(self.cache_file + ".lock")
            with lock:
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self.cache_dict, f)

            self.add_n = 0

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    return cache
                except EOFError:
                    print("Cache file incomplete. Clearing and retrying...")
                    if not allow_retry:
                        raise
                    with open(self.cache_file, "wb") as f:
                        pickle.dump({}, f)
                    return {}
                except Exception:
                    if not allow_retry:
                        raise
                    print("Pickle Error: Retry in 5 seconds...")
                    time.sleep(5)
        return {}

