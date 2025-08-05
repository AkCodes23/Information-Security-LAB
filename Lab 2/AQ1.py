import time
import matplotlib.pyplot as plt
from Crypto.Cipher import DES, AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Messages
messages = [
    b"Message One",
    b"Message Two",
    b"Message Three",
    b"Message Four",
    b"Message Five"
]

# Keys
des_key = b"12345678"
aes_keys = {
    "AES-128": get_random_bytes(16),
    "AES-192": get_random_bytes(24),
    "AES-256": get_random_bytes(32)
}

# Modes
modes = {
    "ECB": AES.MODE_ECB,
    "CBC": AES.MODE_CBC
}

# Function to time encryption + decryption
def time_crypto(algorithm, key, mode, msg, iv=None):
    # Create cipher for encryption
    enc_cipher = algorithm.new(key, mode, iv=iv) if iv else algorithm.new(key, mode)
    start = time.perf_counter()
    ct = enc_cipher.encrypt(pad(msg, enc_cipher.block_size))
    # Create cipher for decryption
    dec_cipher = algorithm.new(key, mode, iv=iv) if iv else algorithm.new(key, mode)
    pt = unpad(dec_cipher.decrypt(ct), dec_cipher.block_size)
    end = time.perf_counter()
    return (end - start) * 1e6  # microseconds

# Store results
results = {mode: {"DES": [], "AES-128": [], "AES-192": [], "AES-256": []} for mode in modes}

# Run tests
for mode_name, mode_type in modes.items():
    for msg in messages:
        # DES
        iv_des = get_random_bytes(8) if mode_type != AES.MODE_ECB else None
        des_time = time_crypto(DES, des_key, mode_type, msg, iv_des)
        results[mode_name]["DES"].append(des_time)

        # AES
        for label, key in aes_keys.items():
            iv_aes = get_random_bytes(16) if mode_type != AES.MODE_ECB else None
            aes_time = time_crypto(AES, key, mode_type, msg, iv_aes)
            results[mode_name][label].append(aes_time)

# Plot graphs
for mode in results:
    plt.figure(figsize=(8, 5))
    for algo in results[mode]:
        plt.plot(range(1, 6), results[mode][algo], marker='o', label=algo)
    plt.title(f"Execution Time in {mode} Mode")
    plt.xlabel("Message Index")
    plt.ylabel("Time (µs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
