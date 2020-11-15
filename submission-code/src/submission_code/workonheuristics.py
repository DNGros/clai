from readdata import get_all_data
import random


def main():
    data = get_all_data(preparse=False)
    random.seed(40)
    print("SDFsadfasdf")
    for i, v in enumerate(random.sample(data.examples, 100)):
        print(i)
        print(v)


if __name__ == "__main__":
    main()
