"""
Download all real fraud detection datasets for STREAM-FraudX experiments.

Prerequisites:
1. Install Kaggle API: conda run -n py310 pip install kaggle
2. Setup Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Create API token (downloads kaggle.json)
   - Place in ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json

3. Accept competition/dataset rules on Kaggle website:
   - IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
   - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
   - Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
"""

import argparse
from pathlib import Path


def download_all_datasets(data_root: str = 'data'):
    """Download all three fraud detection datasets."""

    print("="*80)
    print("STREAM-FraudX Dataset Downloader")
    print("="*80)

    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)

    # 1. IEEE-CIS Fraud Detection
    print("\n[1/3] IEEE-CIS Fraud Detection (Kaggle Competition)")
    print("-"*80)
    try:
        from stream_fraudx.data.ieee_cis_loader import download_ieee_cis
        download_ieee_cis(data_dir=str(data_path / 'ieee-cis'))
        print("✓ IEEE-CIS downloaded successfully")
    except Exception as e:
        print(f"✗ IEEE-CIS download failed: {e}")
        print("  Make sure you've accepted the competition rules!")

    # 2. PaySim Mobile Money
    print("\n[2/3] PaySim Mobile Money Simulator")
    print("-"*80)
    try:
        from stream_fraudx.data.paysim_loader import download_paysim
        download_paysim(data_dir=str(data_path / 'paysim'))
        print("✓ PaySim downloaded successfully")
    except Exception as e:
        print(f"✗ PaySim download failed: {e}")

    # 3. Elliptic Bitcoin
    print("\n[3/3] Elliptic Bitcoin Transactions")
    print("-"*80)
    try:
        from stream_fraudx.data.elliptic_loader import download_elliptic
        download_elliptic(data_dir=str(data_path / 'elliptic'))
        print("✓ Elliptic downloaded successfully")
    except Exception as e:
        print(f"✗ Elliptic download failed: {e}")

    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)

    # Check what was downloaded
    datasets = {
        'IEEE-CIS': data_path / 'ieee-cis' / 'train_transaction.csv',
        'PaySim': data_path / 'paysim' / 'PS_20174392719_1491204439457_log.csv',
        'Elliptic': data_path / 'elliptic' / 'elliptic_txs_features.csv'
    }

    for name, file_path in datasets.items():
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {name:<15} {size_mb:>8.1f} MB")
        else:
            print(f"✗ {name:<15} NOT FOUND")

    print("\nNext steps:")
    print("  1. Run experiments: python run_real_experiments.py")
    print("  2. Or test individual loaders:")
    print("     python stream_fraudx/data/ieee_cis_loader.py")
    print("     python stream_fraudx/data/paysim_loader.py")
    print("     python stream_fraudx/data/elliptic_loader.py")


def verify_kaggle_setup():
    """Verify Kaggle API is properly configured."""
    import os

    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if not kaggle_json.exists():
        print("\n⚠ Kaggle API not configured!")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move downloaded kaggle.json to ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Check permissions
    stat_info = kaggle_json.stat()
    if oct(stat_info.st_mode)[-3:] != '600':
        print(f"\n⚠ Incorrect permissions on {kaggle_json}")
        print("Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    print("✓ Kaggle API configured correctly")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download fraud detection datasets')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify Kaggle setup without downloading')

    args = parser.parse_args()

    # Verify Kaggle setup
    if not verify_kaggle_setup():
        exit(1)

    if args.verify_only:
        print("\n✓ Setup verified. Ready to download datasets.")
        exit(0)

    # Download all datasets
    download_all_datasets(args.data_dir)
