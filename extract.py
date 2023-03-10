
with open("./tran_metrics_smi/proposed/job_source_train.txt.o4153462_4_to_2", "r") as file:
    lines = file.readlines()
    counter = 0
    constant_line = ""
    for line in lines:
        # if "best_acc, best_loss" in lines:
        # if "Es" in line:
        # if "my_wx, my_wy, c_my_bound" in line:
        if "lipschitz_constant:" in line:
            # if "acc_without_tl, loss_without_tl" in line:
            parts = line.split(" ")
            print(parts)
            # print(parts[2][20:])
            # print(parts[4])
            # constant = float(parts[1][14:])
            constant = float(parts[1])
            print(constant)
            constant_line += str(constant) + "\t"
            counter += 1

            if counter == 49:
                # print(constant_line)
                with open("myx.txt", "a") as file:
                    file.write(constant_line + "\n")
                counter = 0
                constant_line = ""



with open("./tran_metrics_smi/proposed/job_source_train.txt.o4153462_4_to_2", "r") as file:
    lines = file.readlines()
    counter = 0
    constant_line = ""
    for line in lines:
        if line.startswith("constant "):
            parts = line.split(" ")
            constant = float(parts[1])
            constant_line += str(constant) + "\t"
            counter += 1

            if counter == 49:
                print(constant_line)
                with open("myconstant.txt", "a") as file:
                    file.write(constant_line + "\n")
                counter = 0
                constant_line = ""

