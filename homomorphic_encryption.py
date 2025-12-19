import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import phe  # pip install phe

def generate_keys():
    print("Generating Paillier keypair...")
    pubkey, privkey = phe.generate_paillier_keypair()
    return pubkey, privkey

def encrypt_dataset(data, pubkey):
    """Encrypts a list of values or list of lists."""
    print("Encrypting data (this may take a while)...")
    start = time.time()
    
    # Helper for single value
    def encrypt_val(val):
        return pubkey.encrypt(val)

    encrypted_data = []
    # If data is a DataFrame/Series, convert to list
    if hasattr(data, 'tolist'): 
        data = data.tolist()
    elif hasattr(data, 'values'):
        data = data.values.tolist()

    # Handling 2D array (Features) vs 1D array (Target)
    if isinstance(data[0], list) or isinstance(data[0], np.ndarray):
        for row in data:
            encrypted_data.append([encrypt_val(x) for x in row])
    else:
        encrypted_data = [encrypt_val(x) for x in data]
        
    print(f"Encryption finished in {time.time() - start:.2f} seconds.")
    return encrypted_data

def decrypt_dataset(data, privkey):
    """Decrypts data using the private key."""
    print("Decrypting data...")
    decrypted_data = []
    
    for item in data:
        if isinstance(item, list):
            decrypted_data.append([privkey.decrypt(x) for x in item])
        else:
            decrypted_data.append(privkey.decrypt(item))
            
    return decrypted_data

if __name__ == "__main__":
    # 1. Load the specific split used in training
    try:
        with open('rus_split.pkl', 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    except FileNotFoundError:
        print("Run training script first to generate data splits.")
        exit()

    # For demonstration, we'll use a small subset because Paillier is slow
    subset_size = 50
    print(f"Using a subset of {subset_size} samples for demonstration...")
    X_demo = X_train[:subset_size]
    y_demo = y_train[:subset_size]
    X_test_demo = X_test[:subset_size]
    y_test_demo = y_test[:subset_size]

    # 2. Key Generation
    pub_key, priv_key = generate_keys()

    # 3. Encryption (Simulation of Secure Storage)
    print("\n--- Phase 1: Encryption ---")
    X_train_enc = encrypt_dataset(X_demo, pub_key)
    y_train_enc = encrypt_dataset(y_demo, pub_key)
    # Save encrypted data (simulated)
    # with open('encrypted_db.pkl', 'wb') as f: pickle.dump((X_train_enc, y_train_enc), f)

    # 4. Decryption (Authorized Access for Training)
    print("\n--- Phase 2: Decryption & Verification ---")
    X_train_dec = decrypt_dataset(X_train_enc, priv_key)
    y_train_dec = decrypt_dataset(y_train_enc, priv_key)
    
    # Verify integrity
    print(f"Original Value (sample): {X_demo.iloc[0].tolist()[0]}")
    print(f"Decrypted Value (sample): {X_train_dec[0][0]}")
    assert np.isclose(X_demo.iloc[0].tolist()[0], X_train_dec[0][0]), "Decryption mismatch!"
    print("Integrity Check: PASSED")

    # 5. Train Model on Decrypted Data (as done in the notebook)
    # The notebook uses specific hyperparameters for the 'Encrypted' section
    print("\n--- Phase 3: Training on Decrypted Data ---")
    gb_model = GradientBoostingClassifier(
        learning_rate=0.004,
        max_depth=83,
        max_features='log2',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=300
    )
    
    gb_model.fit(X_train_dec, y_train_dec)
    
    # Predict on (Decrypted) Test Data
    # In a real scenario, you'd encrypt inputs -> send to server -> server computes (HE) -> return result
    # But standard ML models don't support HE inference directly. 
    # We follow the notebook's flow: validation on standard data.
    preds = gb_model.predict(X_test_demo)
    
    print("\nModel Performance on Decrypted Data:")
    print(f"Accuracy: {accuracy_score(y_test_demo, preds):.2%}")
    print(classification_report(y_test_demo, preds))
