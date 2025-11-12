import numpy as np

#  File paths 
X_TRAIN_FILE = 'X_train_scaled.npy'
X_TEST_FILE = 'X_test_scaled.npy'
Y_TRAIN_FILE = 'y_train_binary.npy'
Y_TEST_FILE = 'y_test_binary.npy'

#  Expected parameters 
EXPECTED_FEATURES = 190
EXPECTED_Y_VALUES = {0, 1} # Only binary labels are expected

def validate_npy_files():
    """Loads and validates the shape, dtype, and content of all prepared arrays."""
    print(" Starting NumPy File Validation ")
    
    #  Load data 
    try:
        X_train = np.load(X_TRAIN_FILE)
        X_test = np.load(X_TEST_FILE)
        y_train = np.load(Y_TRAIN_FILE)
        y_test = np.load(Y_TEST_FILE)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: File not found ({e}). Did you run data_prep.py?")
        return
    
    #  Validate shapes and dtypes 
    print("\n1. Shape and Data Type Check:")
    
    # Check X arrays (features)
    for name, arr in [('X_train', X_train), ('X_test', X_test)]:
        # Check dimensions
        if arr.ndim != 2:
            print(f"X {name}: Incorrect dimensions. Expected 2D, got {arr.ndim}D.")
        
        # Check feature count
        if arr.shape[1] != EXPECTED_FEATURES:
            print(f"X {name}: Feature count mismatch. Expected {EXPECTED_FEATURES}, got {arr.shape[1]}.")
        
        # Check data type
        if not np.issubdtype(arr.dtype, np.floating):
             print(f"X {name}: Incorrect dtype. Expected float, got {arr.dtype}.")
        
        print(f"✅ {name}: Shape {arr.shape}, Dtype {arr.dtype}.")

    # Check Y arrays (Labels)
    for name, arr in [('y_train', y_train), ('y_test', y_test)]:
        # Check for 1D or (N, 1) shape
        # fix: The original check was `arr.ndim != 1 and arr.shape[1] != 1` which is too strict for 1D arrays
        # A 1D array has no arr.shape[1], causing an IndexError
        is_correct_y_shape = arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
        if not is_correct_y_shape: 
             print(f"Warning: {name}: Non-standard shape. Using {arr.shape}.")
        
        # Check data type
        if not np.issubdtype(arr.dtype, np.integer):
             print(f"X {name}: Incorrect dtype. Expected integer, got {arr.dtype}.")
        
        print(f"OK: {name}: Shape {arr.shape}, Dtype {arr.dtype}.")


    # Validate content (scaling and labels) 
    print("\n2. Content and Scaling Check:")

    # Check X sScaling (feature data)
    print(f"X_train Min: {X_train.min():.4f} | Max: {X_train.max():.4f}")
    print(f"X_test Min: {X_test.min():.4f} | Max: {X_test.max():.4f}")
    
    
    # Validation helper function for scaling (the fix)
    def check_scaling(arr: np.ndarray, name: str):
        """Checks if array is scaled properly within [0, 1] and is not collapsed."""
        
        arr_min = arr.min()
        arr_max = arr.max()

        # fix: Use np.isclose for robust floating-point boundary check
        # We need to check if the min is close to or greater than 0 and max is close to or less than 1
        is_min_ok = arr_min >= -1e-6 # Allow tiny negative margin to account for float imprecision near 0
        is_max_ok = arr_max <= 1.0 + 1e-6 # Allow tiny positive margin to account for float imprecision near 1

        is_in_range = is_min_ok and is_max_ok
        
        # check if it has the data collapsed (min == max)?
        # Use np.isclose for float comparison
        is_collapsed = np.isclose(arr_min, arr_max)

        if not is_in_range:
            print(f"X Scaling check FAILED for {name}. Values are outside the expected [0, 1] range. (Min: {arr_min:.6f}, Max: {arr_max:.6f})")
        elif is_collapsed:
            print(f"Warning: Scaling check WARNING for {name}. Data appears to have identical min/max values.")
        else:
            print(f"OK {name} scaling looks correct ([0, 1] range enforced).")

    # Run checks
    check_scaling(X_train, 'X_train')
    check_scaling(X_test, 'X_test')
    
    
    # Check Y labels (binary)
    y_train_unique = np.unique(y_train)
    y_test_unique = np.unique(y_test)

    if set(y_train_unique) != EXPECTED_Y_VALUES or set(y_test_unique) != EXPECTED_Y_VALUES:
        print(f"X Label check FAILED. Expected labels {EXPECTED_Y_VALUES}.")
        print(f"   y_train unique: {y_train_unique}")
        print(f"   y_test unique: {y_test_unique}")
    else:
        print("OK Y label content is correct (contains only 0 and 1).")
    
    # Check class imbalance (important for model training!)
    # We must handle the case where a class might be missing
    
    print(f"\nTraining Class Distribution:")
    train_counts = np.bincount(y_train)
    
    # Safely get counts for 0 and 1
    count_0 = train_counts[0] if len(train_counts) > 0 else 0
    count_1 = train_counts[1] if len(train_counts) > 1 else 0

    print(f" Normal (0): {count_0} samples")
    print(f" Attack (1): {count_1} samples")

    if count_1 == 0:
        print(" Imbalance Ratio: N/A (No Attack samples in training set)")
    else:
        print(f" Imbalance Ratio: {(count_0 / count_1):.2f} (Normal/Attack)")

if __name__ == '__main__':
    validate_npy_files()