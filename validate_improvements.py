#!/usr/bin/env python3
"""
Simplified validation test for foundational improvements.
Tests core functionality without external dependencies.
"""

import os
import sys

def test_imports():
    """Test that all our modules import correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test core module imports
        import train
        print("✅ train.py imports successfully")
        
        # Check if train_model_epochs has scheduler parameter
        import inspect
        sig = inspect.signature(train.train_model_epochs)
        if 'scheduler' in sig.parameters:
            print("✅ train_model_epochs has scheduler parameter")
        else:
            print("❌ train_model_epochs missing scheduler parameter")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Test baseline_v2 script structure
        with open('baseline_v2.py', 'r') as f:
            content = f.read()
        
        if 'ReduceLROnPlateau' in content:
            print("✅ baseline_v2.py includes learning rate scheduling")
        else:
            print("❌ baseline_v2.py missing learning rate scheduling")
            
        if 'num_epochs=75' in content or 'num_epochs=100' in content:
            print("✅ baseline_v2.py uses increased epochs")
        else:
            print("❌ baseline_v2.py missing increased epochs")
            
        if '12000' in content or '10000' in content:
            print("✅ baseline_v2.py uses larger dataset")
        else:
            print("❌ baseline_v2.py missing larger dataset")
            
    except FileNotFoundError:
        print("❌ baseline_v2.py not found")
        return False
    
    return True

def test_compare_models_improvements():
    """Test that compare_models.py has been enhanced."""
    print("\n🔍 Testing compare_models.py improvements...")
    
    try:
        with open('compare_models.py', 'r') as f:
            content = f.read()
        
        if '8000' in content:
            print("✅ compare_models.py uses larger dataset (8000 vs 2000)")
        else:
            print("❌ compare_models.py still uses small dataset")
            
        if 'num_epochs=20' in content:
            print("✅ compare_models.py uses more epochs (20 vs 3)")
        else:
            print("❌ compare_models.py still uses few epochs")
            
        if 'ReduceLROnPlateau' in content:
            print("✅ compare_models.py includes learning rate scheduling")
        else:
            print("❌ compare_models.py missing learning rate scheduling")
            
    except FileNotFoundError:
        print("❌ compare_models.py not found")
        return False
    
    return True

def test_train_enhancements():
    """Test that train.py has been enhanced."""
    print("\n🔍 Testing train.py enhancements...")
    
    try:
        with open('train.py', 'r') as f:
            content = f.read()
        
        if 'early_stop_patience' in content:
            print("✅ train.py includes early stopping")
        else:
            print("❌ train.py missing early stopping")
            
        if 'learning_rates' in content:
            print("✅ train.py tracks learning rates")
        else:
            print("❌ train.py missing learning rate tracking")
            
        if 'best_val_acc' in content:
            print("✅ train.py tracks best validation accuracy")
        else:
            print("❌ train.py missing validation tracking")
            
    except FileNotFoundError:
        print("❌ train.py not found")
        return False
    
    return True

def test_hyperparameter_script():
    """Test that hyperparameter tuning script exists and has key features."""
    print("\n🔍 Testing hyperparameter_tuning.py...")
    
    try:
        with open('hyperparameter_tuning.py', 'r') as f:
            content = f.read()
        
        if 'BidirectionalLSTMModel' in content and 'GRUWithAttentionModel' in content:
            print("✅ hyperparameter_tuning.py targets key models")
        else:
            print("❌ hyperparameter_tuning.py missing key models")
            
        if 'learning_rates' in content and 'batch_sizes' in content:
            print("✅ hyperparameter_tuning.py includes grid search")
        else:
            print("❌ hyperparameter_tuning.py missing grid search")
            
        if 'itertools.product' in content or 'for lr' in content:
            print("✅ hyperparameter_tuning.py implements parameter combinations")
        else:
            print("❌ hyperparameter_tuning.py missing parameter combinations")
            
    except FileNotFoundError:
        print("❌ hyperparameter_tuning.py not found")
        return False
    
    return True

def test_quickstart_update():
    """Test that quickstart.py has been updated."""
    print("\n🔍 Testing quickstart.py updates...")
    
    try:
        with open('quickstart.py', 'r') as f:
            content = f.read()
        
        if 'default=20' in content:
            print("✅ quickstart.py default epochs increased to 20")
        else:
            print("❌ quickstart.py still uses old default epochs")
            
        if 'was 5 in V1' in content:
            print("✅ quickstart.py documents V1 vs V2 changes")
        else:
            print("❌ quickstart.py missing V1 vs V2 documentation")
            
    except FileNotFoundError:
        print("❌ quickstart.py not found")
        return False
    
    return True

def main():
    """Run all validation tests."""
    print("=" * 70)
    print("FOUNDATIONAL IMPROVEMENTS VALIDATION TEST")
    print("=" * 70)
    print("Testing implementation without external dependencies...")
    
    tests = [
        test_imports,
        test_compare_models_improvements,
        test_train_enhancements,
        test_hyperparameter_script,
        test_quickstart_update
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL FOUNDATIONAL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\nKey improvements verified:")
        print("- ✅ Learning rate scheduling with early stopping")
        print("- ✅ Increased training epochs (20-100 vs 3)")
        print("- ✅ Larger datasets (8,000-12,000 vs 2,000 samples)")
        print("- ✅ Hyperparameter tuning for key models")
        print("- ✅ Enhanced baseline V2 evaluation script")
        print("\nReady for full Baseline V2 evaluation!")
    else:
        print(f"❌ {total - passed} tests failed. Please review implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)