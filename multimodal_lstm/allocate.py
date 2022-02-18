import json
import os
import random


def write_to_jsons(file_path, data_list):
    dir_path = file_path.rsplit("/",1)[0]
    print(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "a+") as f1:
        for data in data_list:
            data_json = json.dumps(data, ensure_ascii=False)
            f1.write(data_json)
            f1.write("\n")
        f1.close()


def allocate(root):
    normal_data_list = []
    abnormal_data_list = []
    file_paths = os.listdir(root)
    for fp in file_paths:
        print(fp)
        with open(root+'/'+fp, "r") as f:
            for line in f:
                one_trace = json.loads(line)
                trace_bool = one_trace.get('trace_bool')
                if not trace_bool:
                    abnormal_data_list.append(one_trace)
                else:
                    normal_data_list.append(one_trace)

    write_to_jsons("./test/abnormal/abnormal_test.jsons", abnormal_data_list)

    len_train = int(len(normal_data_list) * 0.6)
    random.shuffle(normal_data_list)
    train_data_list = normal_data_list[:len_train]
    test_normal_data_list = normal_data_list[len_train:len(normal_data_list)]
    write_to_jsons("./train/train.jsons", train_data_list)
    write_to_jsons("./test/normal/normal_test.jsons", test_normal_data_list)


def main():
    allocate("./data")


if __name__ == '__main__':
    main()
