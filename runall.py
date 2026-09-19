import argparse


from data import get_data
from util import run_quantum_experiment, run_classical_baseline, export_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    dataset = get_data(args.dataset, test_size=0.2, seed=args.iteration)
    classical_accuracy, classical_train_s, classical_predict_s = run_classical_baseline(
        dataset,
        max_depth=2,
        random_state=None,
    )
    record = run_quantum_experiment(
        dataset=dataset,
        classical_accuracy=classical_accuracy,
        classical_train_s=classical_train_s,
        classical_predict_s=classical_predict_s,
        batch_size=args.batch_size,
        depth=2,
    )
    file_name = f"results/{args.iteration}_{args.dataset}_{args.batch_size}.csv"
    export_csv([record], file_name)


if __name__ == "__main__":
    main()
