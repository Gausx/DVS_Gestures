import os  # NOQA: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # NOQA: E402
from SNN import SNNs_MLP,SNNs_CNN

import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


logPath = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'log'
if not os.path.exists(logPath):
    os.makedirs(logPath)
sys.stdout = Logger(logPath + os.sep + "log_gesture_SNN.txt")


def main():
    SNNs_MLP.main()
    SNNs_CNN.main()


if __name__ == '__main__':
    main()
