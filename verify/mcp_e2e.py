"""End-to-end MCP protocol test: spawn the real server over stdio and drive it.

This exercises the transport, the initialize handshake and tools/list /
tools/call -- not just the handler function. Run with:

    env -u PYTHONPATH .venv/bin/python verify/mcp_e2e.py --model-dir out/von-1.0-mlx/8bit
"""

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="out/von-1.0-mlx/8bit")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # avoid leaking the parent interpreter's path
    env["PYTHONPATH"] = root

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "von_mlx.mcp_server", "--model-dir", args.model_dir],
        cwd=root,
        env=env,
    )

    failures = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("initialize: OK")

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"tools/list: {names}")
            for expected in ("decide", "judge", "rate", "system_one", "model_info"):
                if expected not in names:
                    failures.append(f"tool {expected} missing from tools/list")

            print("\ntools/call decide ...")
            r = await session.call_tool("decide", {
                "state": "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
                "instructions": "Classify the root cause domain of this incident.",
                "choices": {"infrastructure": "Database, hardware, network failures",
                            "billing": "Invoices, payments, refunds"},
            })
            payload = json.loads(r.content[0].text)
            print("  ", json.dumps(payload))
            if payload.get("choice") != "infrastructure":
                failures.append(f"decide returned {payload.get('choice')!r}")

            print("\ntools/call judge ...")
            r = await session.call_tool("judge", {
                "state": "User cannot log in after the SSO migration.",
                "instructions": "Is the user blocked from the product?",
            })
            payload = json.loads(r.content[0].text)
            print("  ", json.dumps(payload))
            if not (0.0 <= payload.get("probability", -1) <= 1.0):
                failures.append(f"judge probability out of range: {payload}")

            print("\ntools/call model_info ...")
            r = await session.call_tool("model_info", {})
            payload = json.loads(r.content[0].text)
            print("  ", json.dumps(payload))
            if payload.get("backend") != "mlx":
                failures.append("model_info did not report mlx backend")

            print("\ntools/call system_one (fan-out) ...")
            r = await session.call_tool("system_one", {
                "state": "Payment gateway reports timeout on charge authorizations. Urgent.",
                "questions": {
                    "intent": {"type": "choice",
                               "instructions": "Operational nature of this ticket?",
                               "criteria": {"payment_failure": "charge failures",
                                            "access_issue": "login errors"}},
                    "is_urgent": {"type": "noul",
                                  "instructions": "Requires immediate SLA intervention?"},
                },
            })
            payload = json.loads(r.content[0].text)
            print("  ", json.dumps(payload)[:300])
            if len(payload.get("answers", {})) != 2:
                failures.append(f"system_one returned {len(payload.get('answers', {}))} answers")

    print("\n== result ==")
    if failures:
        for f in failures:
            print("  FAIL:", f)
        return 1
    print("  PASS: MCP transport, handshake, tools/list and tools/call all work")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
