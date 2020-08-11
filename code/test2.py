import os  # NOQA: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # NOQA: E402
from LIFLSTM import LIF_LSTM_MLP2,LIF_LSTM_CNN,LIF_GRU_CNN,LIF_RNN_CNN

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
sys.stdout = Logger(logPath + os.sep + "log_gesture_MLP_LSTM_test_ghh2.txt")


def main():
    # print(2)
    # LIF_GRU.main()
    LIF_LSTM_MLP2.main()
    # LIF_LSTM_CNN.main()


if __name__ == '__main__':
    main()
