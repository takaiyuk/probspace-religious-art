import math
import time


def time_since(since: time.time, percent: float) -> str:
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{as_minutes(s)} (remain {as_minutes(rs)})"
