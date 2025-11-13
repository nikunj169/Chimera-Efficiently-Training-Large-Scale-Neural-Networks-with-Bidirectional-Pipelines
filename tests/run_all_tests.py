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
    
    test_files = [
        "test_edge_cases_and_correctness.py",
        "test_integration_complete.py",
        "test_paper_implementation_complete.py",
        "test_person_a_implementation.py",
        "test_person_b_implementation.py",
        "run_validation.py",
    ]
    
    all_passed = True
    results = {}

    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"RUNNING {test_file.upper()}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            [sys.executable, str(test_dir / test_file)],
            capture_output=False # Let the test script print its own output
        )
        
        if result.returncode != 0:
            all_passed = False
            results[test_file] = "FAILED"
        else:
            results[test_file] = "PASSED"
            
    # Summary
    print("\n" + "="*60)
    print("OVERALL TEST SUMMARY")
    print("="*60)
    
    for test_file, status in results.items():
        if status == "PASSED":
            print(f"✓ {test_file}: PASSED")
        else:
            print(f"✗ {test_file}: FAILED")
            
    if all_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
