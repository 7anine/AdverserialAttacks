import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_and_preprocess_unsw_nb15(
    train_file_path: str = 'data/UNSW_NB15_training-set.csv',
    test_file_path: str = 'data/UNSW_NB15_testing-set.csv'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the previously downloaded UNSW-NB15 training and testing datasets, performs necessary
    cleaning, one-hot encoding, feature alignment, and scaling.

    Args:
        train_file_path: Path to the training CSV file.
        test_file_path: Path to the testing CSV file.

    Returns:
        A tuple containing the processed data: 
        (X_train_scaled, X_test_scaled, y_train_binary, y_test_binary)
    """

    print(" Starting Data Preparation for UNSW-NB15 ")
    
    # 1. Load Data
    try:
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)
        print(f"Loaded Training Set Shape: {train_df.shape}")
        print(f"Loaded Testing Set Shape: {test_df.shape}")
    except FileNotFoundError as e:
        print(f"Error: One or more files not found. Ensure they are in the 'data/' folder.")
        print(f"Details: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])


    # Define the categorical features that need One-Hot Encoding
    categorical_features = ['proto', 'service', 'state']
    
    # Define columns to be removed (metadata not useful for NIDS classification)
    cols_to_drop = ['id', 'srcip', 'dstip', 'sport', 'dsport']


    #  Cleaning Functions 
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Applies data cleaning steps to a single DataFrame."""
        
        # Drop irrelevant columns
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Use non-inplace replacement to avoid FutureWarning
        # Replace known special categorical values with a consistent string
        df['service'] = df['service'].replace('-', 'none')
        
        # Replace all NaN/Missing values with 0
        df.fillna(0, inplace=True) # inplace=True is fine here

        # Handle Infinite Values (np.inf, -np.inf)
        for col in df.columns:
            if df[col].dtype != 'object': # Only check numerical columns
                is_finite_mask = np.isfinite(df[col])
                if (~is_finite_mask).any():
                    max_finite = df[col][is_finite_mask].max()
                    df[col].replace([np.inf], max_finite, inplace=True)
                    df[col].replace([-np.inf], 0, inplace=True)
                    
        return df
    
    # Apply cleaning to both datasets
    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)
    print("Data cleaning complete.")


    #  One-Hot Encoding and Alignment 
    
    # 1. Apply One-Hot Encoding
    train_df_encoded = pd.get_dummies(train_df, columns=categorical_features, prefix=categorical_features, drop_first=False)
    test_df_encoded = pd.get_dummies(test_df, columns=categorical_features, prefix=categorical_features, drop_first=False)

    # 2. Align Columns (important, we want both X_train and X_test to have the same features)
    feature_cols = [col for col in train_df_encoded.columns if col not in ['attack_cat', 'label']]

    # Align the test set to the training set features (add missing features as 0s)
    test_df_aligned = test_df_encoded.reindex(columns=feature_cols + ['attack_cat', 'label'], fill_value=0)
    
    # Align the training set so that the order is identical
    train_df_aligned = train_df_encoded.reindex(columns=feature_cols + ['attack_cat', 'label'], fill_value=0)
    
    print(f"Number of final features after encoding/alignment: {len(feature_cols)}")


    #  Separate Features (X) and Labels (y) 
    
    # Binary Classification: y=0 (Normal) or y=1 (Attack)
    y_train_binary = train_df_aligned['label'].values
    y_test_binary = test_df_aligned['label'].values
    
    # X is the feature matrix, containing all columns EXCEPT the target labels.
    X_train = train_df_aligned.drop(columns=['label', 'attack_cat']).values
    X_test = test_df_aligned.drop(columns=['label', 'attack_cat']).values
    
    
    #  Normalize/scale features (MinMaxScaler) 
    scaler = MinMaxScaler()
    
    # 1. FIT the scaler ONLY on the training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 2. TRANSFORM both the training and testing data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)

    print(" Data Preparation Complete ")
    
    return X_train_scaled, X_test_scaled, y_train_binary, y_test_binary


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_unsw_nb15()
    
    if X_train.size > 0:
        print(f"Final X_train shape: {X_train.shape}")
        print(f"Final X_test shape: {X_test.shape}")
        
        #  save the processed arrays to .npy files 
        print("\nSaving final NumPy arrays...")
        
        np.save('X_train_scaled.npy', X_train)
        np.save('X_test_scaled.npy', X_test)
        np.save('y_train_binary.npy', y_train)
        np.save('y_test_binary.npy', y_test)
        
        print("Files saved successfully, ready for model training/evaluation")