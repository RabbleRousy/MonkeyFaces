import time

a = time.perf_counter()
for _ in range(10):
    time.sleep(0.5)

print(time.perf_counter()-a)