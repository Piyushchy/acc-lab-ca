"""Apply the model's recommendation to a real AWS Auto Scaling Group (optional, needs AWS credentials).

    pip install boto3
    python main.py                                         # writes results/latest_recommendation.json
    python cloud/aws_apply.py --asg my-asg --region ap-south-1            # DRY RUN (prints only)
    python cloud/aws_apply.py --asg my-asg --region ap-south-1 --apply    # really sets desired capacity

Also publishes the forecast as a CloudWatch custom metric (--publish-metric) so it can be graphed
next to the actual MemoryUsed metric in the demo. ASG min/max limits still bound the result.
NOTE: written for the lab demo; not exercised against a live account in this repository.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asg", required=True, help="Auto Scaling Group name")
    ap.add_argument("--region", default=None)
    ap.add_argument("--rec", default="results/latest_recommendation.json")
    ap.add_argument("--apply", action="store_true", help="actually call AWS (default is dry run)")
    ap.add_argument("--publish-metric", action="store_true")
    args = ap.parse_args()

    rec = json.load(open(args.rec))
    n = rec["recommended_instances"]
    print(f"forecast ({rec['horizon_min']} min ahead): {rec['predicted_memory_mb']:.0f} MB "
          f"(upper {rec['predicted_upper_mb']:.0f} MB) -> desired capacity {n}")
    if not args.apply:
        print("dry run: nothing changed (add --apply to call AWS)")
        return

    import boto3  # imported lazily so the rest of the repo has no AWS dependency

    asg = boto3.client("autoscaling", region_name=args.region)
    asg.set_desired_capacity(AutoScalingGroupName=args.asg, DesiredCapacity=n, HonorCooldown=True)
    print(f"set desired capacity of {args.asg} to {n}")
    if args.publish_metric:
        cw = boto3.client("cloudwatch", region_name=args.region)
        cw.put_metric_data(Namespace="AccLabCA", MetricData=[
            {"MetricName": "PredictedMemoryMB", "Value": rec["predicted_memory_mb"], "Unit": "Megabytes"},
            {"MetricName": "PredictedUpperMemoryMB", "Value": rec["predicted_upper_mb"], "Unit": "Megabytes"},
            {"MetricName": "RecommendedInstances", "Value": n, "Unit": "Count"},
        ])
        print("published CloudWatch metrics (namespace AccLabCA)")


if __name__ == "__main__":
    main()
