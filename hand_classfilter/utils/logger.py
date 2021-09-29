import sys


class Logger(object):
    def __init__(self, filename="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
