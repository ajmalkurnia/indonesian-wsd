import os
import time

class Log():
    LOG_DIRECTORY = "log/"
    def __init__(self, file_name="log_{}.txt".format(int(time.time())) ):
        if not os.path.exists(self.LOG_DIRECTORY): os.makedirs(self.LOG_DIRECTORY)
        self.log_file = self.LOG_DIRECTORY+file_name

    def write(self, msg):
        print(msg)
        with open (self.log_file, "a") as f:
            f.write(str(msg))
            f.write("\n")

if __name__ == "__main__":
    pass
