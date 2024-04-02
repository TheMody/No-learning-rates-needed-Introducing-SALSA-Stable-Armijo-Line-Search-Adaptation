import sys

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        pass   