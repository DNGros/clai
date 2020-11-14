from pathlib import Path

from evaluate import evaluate_model

cur_file = Path(__file__).parent.absolute()

def main():
    annotation_filepath = cur_file / "../configs/annotations/local_eval_annotations.json"
    params_filepath = cur_file / "../configs/core/evaluation_params.json"
    result = evaluate_model(str(annotation_filepath), str(params_filepath))
    print(result)
    #elif args.mode == 'energy':
    #    result = compute_energyusage(args.annotation_filepath)


if __name__ == '__main__':
    main()
