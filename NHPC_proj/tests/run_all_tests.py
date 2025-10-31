"""
Run all unit and integration tests for Person A and Person B.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all test suites"""
    test_dir = Path(__file__).parent
    
    print("="*60)
    print("Running Chimera Complete Test Suite")
    print("="*60)
    
    # Test Person A
    print("\n" + "="*60)
    print("PERSON A TESTS")
    print("="*60)
    
    result_a = subprocess.run(
        [sys.executable, str(test_dir / "test_person_a_implementation.py")],
        capture_output=False
    )
    
    # Test Person B
    print("\n" + "="*60)
    print("PERSON B TESTS")
    print("="*60)
    
    result_b = subprocess.run(
        [sys.executable, str(test_dir / "test_person_b_implementation.py")],
        capture_output=False
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if result_a.returncode == 0:
        print("✓ Person A tests: PASSED")
    else:
        print("✗ Person A tests: FAILED")
    
    if result_b.returncode == 0:
        print("✓ Person B tests: PASSED")
    else:
        print("✗ Person B tests: FAILED")
    
    # Exit with error if any failed
    if result_a.returncode != 0 or result_b.returncode != 0:
        sys.exit(1)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_tests()
