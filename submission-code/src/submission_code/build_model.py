from irtoy import build_ir_model, cache_model, build_model_all_data


def main():
    cache_model(build_model_all_data())


if __name__ == "__main__":
    main()