#!/usr/bin/env python3
"""
Verda Dashboard Launcher v2.5.0 - Robust startup with fail-safes.

Features:
- Auto port switching if port is in use
- Multiple fallback port strategies
- Retry logic with exponential backoff
- Graceful error handling and recovery

Run: python run_dashboard.py
"""

import os
import sys

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

def install_dependencies():
    """Install missing dependencies."""
    import subprocess
    packages = ['fastapi', 'uvicorn', 'websockets', 'pydantic']
    print("📦 Installing missing dependencies...")
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
            print(f"   ✅ {pkg}")
        except Exception as e:
            print(f"   ❌ {pkg}: {e}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🚀 Verda Dashboard Launcher v2.5.0                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Try to import and run
    try:
        from verda_mcp.api_server import quick_start
        quick_start()
    except ImportError as e:
        print(f"⚠️  Import error: {e}")

        # Offer to install dependencies
        try:
            response = input("\n📦 Install missing dependencies? (y/n): ").strip().lower()
            if response == 'y':
                install_dependencies()
                print("\n🔄 Retrying...")
                from verda_mcp.api_server import quick_start
                quick_start()
            else:
                print("\nManual install: pip install fastapi uvicorn websockets pydantic")
        except Exception:
            print("\nManual install: pip install fastapi uvicorn websockets pydantic")

    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

        # Offer retry
        try:
            response = input("\n🔄 Retry? (y/n): ").strip().lower()
            if response == 'y':
                main()
        except Exception:
            pass

if __name__ == "__main__":
    main()
