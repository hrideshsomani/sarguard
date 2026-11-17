"""
Continuous Live Stream Client for AML Surveillance & MLOps Simulation
Uploads / streams 50 transactions every 10 seconds to the AML MLOps Engine.
Each 10-second tick:
1. Evaluates all 50 transactions for SAR red flags (Risk Score 0-100 & Typologies).
2. Continually trains / adapts the hybrid model ensemble with anti-forgetting memory replay.
3. Prints real-time MLOps telemetry and flagged incidents to the console.
"""

import sys
import time
import json
import urllib.request
import urllib.error

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

API_URL = "http://127.0.0.1:8000/simulation/tick"
INTERVAL_SECONDS = 10
BATCH_SIZE = 50

def run_simulation(cycles=None):
    print("=" * 80)
    print("      AML MLOPS CONTINUOUS LIVE STREAMING SIMULATOR (50 txns / 10s)")
    print("=" * 80)
    print(f"  * Target Endpoint:      {API_URL}")
    print(f"  * Batch Size:           {BATCH_SIZE} transactions per tick")
    print(f"  * Streaming Interval:   {INTERVAL_SECONDS} seconds (5.0 txns/sec)")
    print("  * Anti-Forgetting:      Stratified Experience Replay Active")
    print("=" * 80 + "\n")
    print("[+] Starting live simulation. Press Ctrl+C to stop.\n")

    cycle = 0
    total_streamed = 0
    total_flagged = 0

    while True:
        cycle += 1
        payload = {
            "batch_size": BATCH_SIZE,
            "customer_id": "ALL",
            "threshold": 0.35,
            "auto_train": True
        }

        try:
            t0 = time.time()
            req = urllib.request.Request(
                API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode())
                elapsed = time.time() - t0

            batch_flagged = res["sar_flagged_count"]
            total_streamed += res["batch_size"]
            total_flagged += batch_flagged
            mlops = res["mlops_training_status"]

            print("-" * 80)
            print(f"⚡ TICK #{cycle:04d} | {res['timestamp']} | Latency: {elapsed:.2f}s")
            print(f"   • Batch Streamed:    {res['batch_size']} transactions (Rate: {res['batch_size']/INTERVAL_SECONDS:.1f} txns/sec)")
            print(f"   • SAR Intercepted:   {batch_flagged} flagged violations ({res['sar_flagged_ratio']*100:.1f}%)")
            print(f"   • Total Cumulative:  {total_streamed:,} screened | {total_flagged:,} total SAR alerts")
            print(f"   • Model Accuracy:    {mlops['current_accuracy']*100:.2f}% | ROC-AUC: {mlops['current_roc_auc']:.4f}")
            print(f"   • Memory Reservoir:  {mlops['exemplars_in_memory']:,} exemplars | Retrain Cycles: #{mlops['continuous_learning_cycles']}")

            sample = res.get("flagged_sample", [])
            if sample:
                print("   • Sample Intercepted Red Flags:")
                for item in sample[:3]:
                    print(f"     [!] {item['originator_name']} (${item['amount']:,.2f}) -> {item['typology'][:35]} | Risk: {item['sar_risk_score']:.1f}")

        except urllib.error.URLError as e:
            print(f"[!] Simulation tick #{cycle} connection error: {e}. Retrying in {INTERVAL_SECONDS}s...")
        except Exception as ex:
            print(f"[!] Tick exception: {ex}")

        if cycles and cycle >= cycles:
            print(f"\n[OK] Completed {cycles} simulation cycles.")
            break

        print(f"\n[⏳] Next 50-transaction ingestion scheduled in {INTERVAL_SECONDS} seconds...")
        try:
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n[!] Simulation paused by user. Cleanly exiting.")
            break

if __name__ == "__main__":
    max_c = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_simulation(max_c)
