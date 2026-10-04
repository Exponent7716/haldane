"""Command-line demo: ``python -m euvsim [--high-na] [--dose D] [--pitch P]``."""
import argparse

from .scanner import EUVScanner


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end EUV scanner simulation")
    ap.add_argument("--high-na", action="store_true", help="EXE-class NA 0.55 anamorphic tool")
    ap.add_argument("--dose", type=float, default=30.0, help="dose for throughput (mJ/cm2)")
    ap.add_argument("--pitch", type=float, default=None, help="line/space pitch to print (nm)")
    a = ap.parse_args()
    s = EUVScanner.high_na() if a.high_na else EUVScanner()
    print(s.report(a.dose))
    pitch = a.pitch or (20.0 if a.high_na else 32.0)
    r = s.print_lines(pitch, pitch / 2)
    print(f"  print {pitch:.0f} nm pitch L/S (k1={r['k1']:.2f}): dose-to-size "
          f"{r['dose_mJ_cm2']:.1f} mJ/cm2, NILS {r['image_nils']:.2f}, CD {r['cd_nm']:.2f} nm, "
          f"LER(3s) {r['ler_3sigma_nm']:.2f} nm, LWR(3s) {r['lwr_3sigma_nm']:.2f} nm")


if __name__ == "__main__":
    main()
