"""
Command-Line Interface & Primary Controller
Usage:
    python main.py --help
    python main.py --benchmark
    python main.py --generate-sar CUST-0281
    python main.py --gui desktop
    python main.py --gui web
"""

import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import SARPipeline

def main():
    parser = argparse.ArgumentParser(description="Enterprise AML Compliance Suite - SAR Narrative Generator")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-model evaluation and print Accuracy, FPR, FNR, ROC-AUC table")
    parser.add_argument("--generate-sar", type=str, metavar="CUSTOMER_ID", help="Generate 5-part SAR narrative and audit seal for specified customer ID")
    parser.add_argument("--masked", action="store_true", help="Apply PII obfuscation when generating SAR")
    parser.add_argument("--gui", choices=["desktop", "web"], help="Launch interactive graphical interface ('desktop' for Tkinter or 'web' for Streamlit)")
    parser.add_argument("--threshold", type=float, default=0.45, help="AML model decision threshold (default 0.45)")
    
    args = parser.parse_args()
    
    if args.gui == "desktop":
        from src.ui.desktop_app import launch_desktop_gui
        launch_desktop_gui()
        return
    elif args.gui == "web":
        import subprocess
        target = os.path.join(PROJECT_ROOT, "src", "ui", "web_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", target])
        return

    pipe = SARPipeline()
    pipe.initialize_data_and_models()

    if args.benchmark or (not args.generate_sar and not args.gui):
        print("\n" + "="*80)
        print("AML MODEL PERFORMANCE BENCHMARKS (Accuracy, FPR, FNR, ROC-AUC)")
        print("="*80)
        metrics = pipe.get_evaluation_metrics(threshold=args.threshold)
        print(metrics['summary_table'].to_string(index=False))
        print("\n[+] Total Flagged High-Risk Cases in Test Stream:")
        cases = pipe.get_flagged_cases()
        print(f"    Found {len(cases)} suspicious subjects requiring SAR filing.")
        print("="*80)

    if args.generate_sar:
        cid = args.generate_sar
        print(f"\n[+] Synthesizing FinCEN / FCA Compliant SAR Narrative for Subject: {cid}...")
        res = pipe.generate_case_sar_and_audit(cid, is_masked=args.masked)
        pkg = res['narrative_package']
        audit = res['audit_record']
        
        print("\n" + "#"*80)
        print("OFFICIAL REGULATORY SAR DOSSIER")
        print("#"*80)
        print(pkg['narrative_text'])
        print("\n" + "-"*80)
        print("CRYPTOGRAPHIC AUDIT SEAL (SHA-256):")
        print(f"  {audit['cryptographic_seal_sha256']}")
        print(f"  Timestamp: {audit['audit_timestamp']}")
        print(f"  Status: {audit['filing_status']}")
        print("-"*80)

if __name__ == '__main__':
    main()
