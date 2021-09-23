import os
import re
​
​
def main():
    directory = os.path.join(os.path.expanduser("~"),
                             "Elasticity/Guo/scaling-convergence/log")
    loss_filename = "loss.csv"
    with open(loss_filename, "w") as loss_file:
        loss_file.write("rank,epoch,step,loss\n")
​
        for filename in os.scandir(directory):
            if filename.is_file():
                if "stdout.log" in filename.path:
                    with open(filename.path, "r") as fil:
                        lines = fil.readlines()
                        rank = -1
                        for line in lines:
                            m = re.match("^kungfu rank=*(\d)", line)
                            if m:
                                rank = int(m.group(1))
                            m = re.match("^epoch: *(\d+) step: *(\d+), loss is *(\d[.]\d+)", line)
                            if m:
                                epoch = int(m.group(1))
                                step = int(m.group(2))
                                loss = float(m.group(3))
                                loss_file.write("{},{},{},{}\n".format(rank,epoch,step,loss))
​
​
if __name__ == "__main__":
    main()
