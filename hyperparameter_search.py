import os
import subprocess
# for p0 in [0.01]:#0.01, 0.1, 1.0, 100.0]:
for p0 in [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]:#0.01, 0.1, 1.0, 100.0]:
    print(p0)
    commands = []
    for q0 in [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]:#0.01, 0.1, 1.0, 100.0]:
    # for q0 in [0.000000001]:#0.01, 0.1, 1.0, 100.0]:
        print(q0)
        for i in range(-5,5):
        # for i in range(3,5):
            command = " python main.py {} {} {}".format(p0, q0, i)
            commands.append(command)
        
    # start all programs
    processes = [subprocess.Popen(program,shell=True) for program in commands]
    # wait
    for process in processes:
        process.wait()
