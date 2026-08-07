"""Learned-group ablation that removes static and dynamic Div scoring."""
from ablation_sa import build_parser, run_ablation


def main() -> None:
    args = build_parser("Learned-group ablation without Div").parse_args()
    # Regression feature order: class-wise standardized SA, DDS.
    run_ablation(args, active=("sa", "dds"), mode="ablation_div")


if __name__ == "__main__":
    main()
