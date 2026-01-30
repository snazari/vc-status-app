#!/usr/bin/env python3
"""
Basic test script to verify all modules load correctly
"""

import sys

def test_imports():
    """Test that all modules can be imported"""
    try:
        # Test utility imports
        print("Testing utility imports...")
        from utils.auth import hash_password, verify_credentials
        from utils.data_processing import process_portfolio_csv, calculate_portfolio_metrics
        from utils.visualizations import create_kpi_gauge
        print("✓ Utility imports successful")
        
        # Test page imports (just check they exist)
        print("\nTesting page files...")
        import os
        pages = ['overview.py', 'finances.py', 'key_decisions.py', 'portfolio.py', 'market_conditions.py', 'data_upload.py']
        for page in pages:
            path = os.path.join('pages', page)
            if os.path.exists(path):
                print(f"✓ {page} exists")
            else:
                print(f"✗ {page} missing")
                return False
        
        # Test main app
        print("\nTesting main app file...")
        if os.path.exists('app.py'):
            print("✓ app.py exists")
        else:
            print("✗ app.py missing")
            return False
            
        print("\n✅ All tests passed! The app structure is complete.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 