"""Command-line entry point for von-mlx."""

import argparse
import json
import os
import sys

DEFAULT_DIR = os.environ.get("VON_MLX_MODEL", os.path.expanduser("~/models/von-1.0-mlx"))


def _load_engine(model_dir, temperature=None):
    from von_mlx.decision import VonEngine

    return VonEngine(model_dir, temperature=temperature)


def cmd_serve(args):
    import uvicorn

    from von_mlx.server import create_app

    app = create_app(args.model_dir, temperature=args.temperature)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_decide(args):
    engine = _load_engine(args.model_dir, args.temperature)
    payload = json.loads(args.problem) if args.problem else json.load(sys.stdin)
    resp = engine.evaluate(state=payload["state"], questions=payload["questions"])
    print(resp.model_dump_json(indent=2))


def cmd_bench(args):
    import time

    import numpy as np

    from von_mlx.types import Choice, Noul, Score

    engine = _load_engine(args.model_dir, args.temperature)
    state = ("Payment gateway reports timeout on charge authorizations. "
             "Customer tier: enterprise. Ticket INC-4091. Urgent.")
    cases = {
        "choice": Choice(
            instructions="What is the operational nature of this ticket?",
            criteria={
                "payment_failure": "Failures processing charges, gateway timeouts, declines",
                "access_issue": "Login, SSO, authentication, or permission errors",
            },
        ),
        "noul": Noul(instructions="Does this require immediate SLA intervention?"),
        "score": Score(
            instructions="Rate the incident severity.",
            criteria=["Low", "Medium", "High", "Critical"],
        ),
    }

    # warmup
    engine.evaluate(state=state, questions=cases)
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        engine.evaluate(state=state, questions=cases)
        times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    resp = engine.evaluate(state=state, questions=cases)
    one = (time.perf_counter() - t0) * 1000

    print(f"von-mlx benchmark  ({args.repeats} repeats, 3 questions, K=2/2/4)")
    print(f"  full system_one call : median {np.median(times):8.2f} ms   "
          f"min {min(times):.2f}  max {max(times):.2f}")
    print(f"  reported total       : {sum(times)/len(times):.2f} ms mean")
    print(f"  single-pass (K total): {one:.2f} ms")
    print(f"  answers: {json.dumps(json.loads(resp.model_dump_json())['answers'])}")


def cmd_verify(args):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, here)
    from verify.parity_check import main as parity_main

    sys.argv = ["parity_check", "--model-dir", args.model_dir]
    return parity_main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="von-mlx", description="MLX Von decision model")
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--temperature", type=float, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("decide", help="run one decision problem (JSON arg or stdin)")
    d.add_argument("--problem", default=None)
    d.set_defaults(func=cmd_decide)

    b = sub.add_parser("bench", help="latency benchmark")
    b.add_argument("--repeats", type=int, default=20)
    b.set_defaults(func=cmd_bench)

    v = sub.add_parser("verify", help="torch-vs-MLX numeric parity check")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
