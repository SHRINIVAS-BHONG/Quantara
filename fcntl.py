# Mock fcntl for Windows to silence hermes-agent warnings
def flock(fd, operation):
    pass

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8
