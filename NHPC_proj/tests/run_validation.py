"""
Quick validation script to check all implementations work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_imports():
    """Validate all imports work"""
    print("Validating imports...")
    
    try:
        from chimera.engine import partition
        print("✓ partition.py")
    except Exception as e:
        print(f"✗ partition.py: {e}")
        return False
    
    try:
        from chimera.engine import schedule
        print("✓ schedule.py")
    except Exception as e:
        print(f"✗ schedule.py: {e}")
        return False
    
    try:
        from chimera.engine import runtime
        print("✓ runtime.py")
    except Exception as e:
        print(f"✗ runtime.py: {e}")
        return False
    
    try:
        from chimera.engine import recompute
        print("✓ recompute.py")
    except Exception as e:
        print(f"✗ recompute.py: {e}")
        return False
    
    try:
        from chimera.models import bert48
        print("✓ bert48.py")
    except Exception as e:
        print(f"✗ bert48.py: {e}")
        return False
    
    try:
        from chimera.models import gpt2_64
        print("✓ gpt2_64.py")
    except Exception as e:
        print(f"✗ gpt2_64.py: {e}")
        return False
    
    print("\nAll imports successful!")
    return True

if __name__ == "__main__":
    validate_imports()
