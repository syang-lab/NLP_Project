import re
from os.path import exists


def data_clean(input_file_name, output_file_name):
    """"
    clean domain adaption file
    1.load dataset
    2.get rid of space lines
    3.lower case
    4.remove unicode
    """
    try:
        with open(input_file_name, "r") as input_raw:
            output_content = list()
            for line in input_raw:
                if not line.isspace():
                    line = re.sub(r"[?|!+?|:|(|)]|\\|-|/.*?/|http\S+|www\S+", "", line.lower())
                    output_content.append(line)
    except FileNotFoundError as error:
        msg = "Sorry, the file" + input_file_name + "does not exist."
        print(msg)
        print("error:" + error)

    if exists(output_file_name):
        return
    else:
        with open(output_file_name, "w") as input_cleaned:
            input_cleaned.writelines(output_content)


if __name__ == "__main__":
    input_file_name = "../data/raw_EDT/Domain_adapation/train_test.txt"
    output_file_name = "../data/raw_EDT/Domain_adapation/train_test_cleaned.txt"

    data_clean(input_file_name, output_file_name)
