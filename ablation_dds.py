"""Learned-group ablation without DDS (named SV in the paper)."""
from ablation_sa import build_parser, run_ablation


def main() -> None:
    args = build_parser("Learned-group ablation without SV/DDS").parse_args()
    # Regression feature order: class-wise standardized SA, Div.
    run_ablation(args, active=("sa", "div"), mode="ablation_sv")


if __name__ == "__main__":
    main()
