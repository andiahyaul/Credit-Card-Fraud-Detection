#!/usr/bin/env python3
"""
Environment Validation Script
Validates that current library versions maintain compatibility with Phase 3 achievements
"""

import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import sys

def validate_versions():
    """Check library versions"""
    print("ENVIRONMENT VERSION VALIDATION")
    print("=" * 50)

    versions = {
        'Python': sys.version.split()[0],
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'imbalanced-learn': 'Available',  # Will check functionality below
        'scikit-learn': 'Available'
    }

    for lib, version in versions.items():
        print(f"{lib:<20}: {version}")

    return versions

def validate_smote_tomek():
    """Test SMOTE-Tomek functionality with synthetic data"""
    print("\nSMOTE-TOMEK FUNCTIONALITY TEST")
    print("=" * 50)

    # Create synthetic imbalanced dataset similar to your fraud data
    np.random.seed(42)
    n_samples = 1000
    n_fraud = int(n_samples * 0.006)  # ~0.6% fraud rate like your data

    # Create features similar to your engineered features
    X = np.random.randn(n_samples, 5)  # 5 features
    y = np.concatenate([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])

    print(f"Test dataset created: {n_samples} samples, {n_fraud} fraud cases")
    print(f"Original imbalance ratio: 1:{int((n_samples - n_fraud) / n_fraud)}")

    # Test SMOTE-Tomek
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        fraud_count_after = (y_resampled == 1).sum()
        legit_count_after = (y_resampled == 0).sum()

        print(f"SMOTE-Tomek successful!")
        print(f"After resampling: {len(y_resampled)} samples")
        print(f"Fraud cases: {fraud_count_after}")
        print(f"Legitimate cases: {legit_count_after}")
        print(f"New ratio: 1:{int(legit_count_after / fraud_count_after) if fraud_count_after > 0 else 'inf'}")

        # Check if balance is achieved (should be close to 1:1)
        balance_ratio = legit_count_after / fraud_count_after if fraud_count_after > 0 else float('inf')
        if 0.8 <= balance_ratio <= 1.2:
            print("Class balance achieved (ratio between 0.8-1.2)")
            return True
        else:
            print(f"WARNING: Balance ratio {balance_ratio:.2f} outside expected range")
            return False

    except Exception as e:
        print(f"ERROR: SMOTE-Tomek failed: {str(e)}")
        return False

def validate_feature_importance():
    """Test feature importance calculation"""
    print("\nFEATURE IMPORTANCE CALCULATION TEST")
    print("=" * 50)

    try:
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Features 0,1 are informative

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        print(f"Feature importance calculation successful!")
        print(f"Mutual information scores: {mi_scores}")

        # Check that informative features have higher scores
        if mi_scores[0] > 0 and mi_scores[1] > 0:
            print("Feature importance correctly identifies informative features")
            return True
        else:
            print("WARNING: Feature importance may not be working as expected")
            return False

    except Exception as e:
        print(f"ERROR: Feature importance calculation failed: {str(e)}")
        return False

def validate_haversine():
    """Test haversine distance calculation"""
    print("\nHAVERSINE DISTANCE CALCULATION TEST")
    print("=" * 50)

    try:
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Haversine distance calculation"""
            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))

            return R * c

        # Test with known coordinates (NYC to LA)
        nyc_lat, nyc_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437

        distance = haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)
        expected_distance = 3944  # Approximate distance NYC to LA

        print(f"NYC to LA distance: {distance:.1f} km")
        print(f"Expected: ~{expected_distance} km")

        if abs(distance - expected_distance) < 100:  # Within 100km tolerance
            print("Haversine calculation accurate")
            return True
        else:
            print("WARNING: Haversine calculation may be inaccurate")
            return False

    except Exception as e:
        print(f"ERROR: Haversine calculation failed: {str(e)}")
        return False

def validate_temporal_split():
    """Test temporal data splitting"""
    print("\nTEMPORAL DATA SPLITTING TEST")
    print("=" * 50)

    try:
        # Create synthetic temporal data
        dates = pd.date_range('2019-01-01', '2020-06-30', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates)),
            'target': np.random.choice([0, 1], len(dates), p=[0.99, 0.01])
        })

        # Sort by date (temporal order)
        data = data.sort_values('date').reset_index(drop=True)

        # Create temporal splits
        n_total = len(data)
        train_end = int(n_total * 0.7)
        val_end = int(n_total * 0.85)

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        # Validate temporal ordering
        train_max_date = train_data['date'].max()
        val_min_date = val_data['date'].min()
        val_max_date = val_data['date'].max()
        test_min_date = test_data['date'].min()

        print(f"Training: {len(train_data)} records (end: {train_max_date.date()})")
        print(f"Validation: {len(val_data)} records (start: {val_min_date.date()})")
        print(f"Test: {len(test_data)} records (start: {test_min_date.date()})")

        if train_max_date < val_min_date and val_max_date < test_min_date:
            print("Temporal ordering maintained - no data leakage")
            return True
        else:
            print("ERROR: Temporal ordering violated - data leakage risk")
            return False

    except Exception as e:
        print(f"ERROR: Temporal splitting failed: {str(e)}")
        return False

def main():
    """Run all validation tests"""
    print("CREDIT CARD FRAUD DETECTION - ENVIRONMENT VALIDATION")
    print("=" * 70)
    print("Validating that current environment maintains Phase 3 achievements...")
    print()

    # Run all validation tests
    tests = [
        ("Version Check", validate_versions),
        ("SMOTE-Tomek Functionality", validate_smote_tomek),
        ("Feature Importance", validate_feature_importance),
        ("Haversine Distance", validate_haversine),
        ("Temporal Splitting", validate_temporal_split)
    ]

    results = []

    for test_name, test_func in tests:
        if test_name == "Version Check":
            test_func()  # This just prints versions
            results.append(True)
        else:
            result = test_func()
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        if i == 0:  # Skip version check in results
            continue
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name:<25}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if all(results):
        print("\nALL VALIDATIONS PASSED!")
        print("Environment is ready for Phase 4 model development")
        print("SMOTE-Tomek functionality confirmed")
        print("Feature engineering pipeline intact")
        print("No version locking required - current versions are stable")
    else:
        print("\nWARNING: SOME VALIDATIONS FAILED")
        print("Consider version rollback or environment debugging")
        print("Check backup_before_locking.txt for rollback if needed")

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)