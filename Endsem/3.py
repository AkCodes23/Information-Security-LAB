# PYTHON SCRIPT (Version 5 - Combined and Expanded)
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
    ICT3141 ULTIMATE EXAM SCRIPT - ALL SCENARIOS V5
    
    Part 1: 12 Base 3-Algorithm Systems (Systems 1-12)
    Part 2: Interactive Classical Ciphers (Systems 13-20)
    Part 3: Exam-Specific & New Combinations (Systems 21+)
    
    All with Menu-Driven Interfaces and Performance Graphs
═══════════════════════════════════════════════════════════════════════════
"""

# --- Imports and Library Checks (Same as V4 script) ---
import sys
import random
import time
import hashlib
import base64
import json
from datetime import datetime
from collections import defaultdict
import math
import os # For clear_screen

print("\n" + "="*70)
print("  ICT3141 ULTIMATE EXAM SCRIPT V5 - INITIALIZING")
print("="*70)

# --- PyCryptodome (Essential) ---
try:
    from Crypto.Cipher import DES, DES3, AES, PKCS1_OAEP
    from Crypto.PublicKey import RSA, ElGamal
    from Crypto.Hash import SHA256, SHA512, SHA1, MD5
    from Crypto.Util import number
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Random import get_random_bytes
    from Crypto.Signature import pkcs1_15
    HAS_CRYPTO = True
    print("  ✓ PyCryptodome loaded")
except ImportError:
    HAS_CRYPTO = False
    print("  ✗ PyCryptodome not installed! (pip install pycryptodome)")
    # sys.exit(1) # Allow running without it for classical ciphers

# --- NumPy (for Hill Cipher & Matrix Ops) ---
try:
    import numpy as np
    HAS_NUMPY = True
    print("  ✓ NumPy loaded")
except ImportError:
    HAS_NUMPY = False
    print("  ⚠ NumPy not available (Hill Cipher disabled)")

# --- PHE (for Paillier Homomorphic Encryption) ---
try:
    from phe import paillier
    HAS_PAILLIER = True
    print("  ✓ Paillier (phe) loaded")
except ImportError:
    HAS_PAILLIER = False
    print("  ⚠ Paillier (phe) not available (Homomorphic systems disabled)")

# --- Matplotlib (Optional, for graphical plots) ---
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("  ✓ Matplotlib loaded")
except ImportError:
    HAS_MATPLOTLIB = False
    print("  ⚠ Matplotlib not available (Using ASCII graphs)")

print("="*70 + "\n")


# --- Utility Functions (PerformanceTracker, gcd, mod_inverse, matrix_mod_inv, ElGamal Signatures - Same as V4 script) ---
class PerformanceTracker:
    """Track and visualize performance metrics with enhanced graphing"""
    # (Code from V4 - including matplotlib option)
    def __init__(self):
        self.metrics = []

    def record(self, operation, time_taken, data_size=0):
        self.metrics.append({
            'operation': operation, 'time': time_taken,
            'size': data_size, 'timestamp': datetime.now().isoformat()
        })

    def get_stats(self, operation=None):
        data = [m for m in self.metrics if not operation or m['operation'] == operation]
        if not data: return None
        times = [m['time'] for m in data]
        sizes = [m['size'] for m in data if m['size'] > 0]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        return {
            'count': len(times), 'average': sum(times) / len(times),
            'min': min(times), 'max': max(times), 'total': sum(times),
            'avg_size': avg_size
        }

    def print_graph(self, title="Performance Analysis"):
        print(f"\n{'='*70}\n  📊 {title}\n{'='*70}")
        ops = defaultdict(list); [ops[m['operation']].append(m['time']) for m in self.metrics]
        if not ops: print("  No data recorded yet"); print("="*70); return

        print("\nOperation Statistics:")
        stats_list = []
        for op, times in sorted(ops.items()):
            avg = sum(times) / len(times)
            stats_list.append({'op': op, 'avg': avg, 'count': len(times), 'min': min(times), 'max': max(times)})
            print(f"\n  {op}:\n    Count:   {len(times)}\n    Average: {avg:.6f}s\n    Min:     {min(times):.6f}s\n    Max:     {max(times):.6f}s")
            scale = 2000; bar = '█' * int(avg * scale); bar = bar if bar else ('·' if avg > 0 else '')
            print(f"    Graph:   {bar}")
        print("\n" + "="*70)

        if HAS_MATPLOTLIB:
            try:
                labels = [s['op'] for s in stats_list]; averages = [s['avg'] for s in stats_list]
                plt.figure(figsize=(10, max(6, len(labels)*0.5))); plt.barh(labels, averages, color='skyblue')
                plt.xlabel("Average Time (seconds)"); plt.ylabel("Operation"); plt.title(title)
                plt.tight_layout(); print("\n📈 Displaying graphical performance chart..."); plt.show()
            except Exception as e: print(f"  ⚠ Could not generate Matplotlib graph: {e}")

    def compare_algorithms(self, alg_list):
        if len(alg_list) < 2: print("Need at least two algorithms to compare."); return
        stats = {alg: s['average'] for alg in alg_list if (s := self.get_stats(alg))}
        if len(stats) < 2: print("Not enough data for comparison."); return

        print(f"\n{'='*70}\n  🚀 ALGORITHM COMPARISON\n{'='*70}")
        sorted_algs = sorted(stats.items(), key=lambda x: x[1])
        print("\nSpeed Ranking (fastest to slowest):")
        scale = 5000; max_len = max(len(alg) for alg, t in sorted_algs)
        for i, (alg, time_val) in enumerate(sorted_algs, 1):
            bar = '█' * int(time_val * scale); bar = bar if bar else ('·' if time_val > 0 else '')
            print(f"  {i}. {alg:<{max_len}s}: {time_val:.8f}s\n     {bar}")

        fastest_time = sorted_algs[0][1]
        if fastest_time > 0:
             print("\nRelative Performance:")
             for alg, time_val in sorted_algs[1:]:
                 print(f"  {alg} is {time_val / fastest_time:.1f}x slower than {sorted_algs[0][0]}")
        else: print("\nFastest algorithm too fast for relative comparison.")
        print("="*70)

        if HAS_MATPLOTLIB:
             try:
                 labels = [a[0] for a in sorted_algs]; times = [a[1] for a in sorted_algs]
                 plt.figure(figsize=(10, max(6, len(labels)*0.5))); plt.barh(labels, times, color='lightcoral')
                 plt.xlabel("Average Time (seconds)"); plt.ylabel("Algorithm"); plt.title("Algorithm Speed Comparison")
                 plt.tight_layout(); print("\n📈 Displaying graphical comparison chart..."); plt.show()
             except Exception as e: print(f"  ⚠ Could not generate Matplotlib comparison graph: {e}")


def gcd(a, b): return math.gcd(a, b) # Use math.gcd

def mod_inverse(a, m):
    # (Code from V4)
    a = a % m
    if gcd(a, m) != 1: return None
    try: return pow(a, -1, m)
    except ValueError: pass
    m0, x0, x1 = m, 0, 1
    while a > 1: q = a // m; m, a = a % m, m; x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += m0
    return x1

def matrix_mod_inv(matrix, modulus):
    # (Code from V4) - Requires NumPy
    if not HAS_NUMPY: raise ImportError("NumPy required")
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = mod_inverse(det % modulus, modulus)
    if det_inv is None: raise ValueError(f"Matrix determinant {det} not invertible mod {modulus}")
    adj = np.linalg.inv(matrix) * det; inv = (det_inv * np.round(adj)) % modulus
    return inv.astype(int)

# --- Manual ElGamal Signature Implementation (Same as V4 script) ---
def generate_elgamal_sig_keys(bits=1024):
    # (Code from V4)
    print(f"  Generating ElGamal signature parameters ({bits} bits)...")
    start_g = time.time()
    while True: q = number.getPrime(bits - 1); p = 2 * q + 1;
        if number.isPrime(p): break
    print(f"    Found safe prime p (took {time.time()-start_g:.2f}s)")
    while True: g_cand = number.getRandomRange(2, p - 1);
        if pow(g_cand, q, p) == 1 and pow(g_cand, 2, p) != 1: g = g_cand; break
    print(f"    Found generator g")
    x = number.getRandomRange(2, q); y = pow(g, x, p)
    print(f"    Generated keys x, y (Total time: {time.time()-start_g:.2f}s)")
    return {'p': p, 'q': q, 'g': g, 'x': x, 'y': y}

def elgamal_sign(msg_bytes, priv_key):
    # (Code from V4)
    p, q, g, x = priv_key['p'], priv_key['q'], priv_key['g'], priv_key['x']
    h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    while True:
        k = number.getRandomRange(2, q); r = pow(g, k, p)
        if r == 0: continue
        k_inv = number.inverse(k, q); s = (k_inv * (h_int + x * r)) % q
        if s == 0: continue
        return int(r), int(s)

def elgamal_verify(msg_bytes, signature, pub_key):
    # (Code from V4)
    p, q, g, y = pub_key['p'], pub_key['q'], pub_key['g'], pub_key['y']
    r, s = signature
    if not (0 < r < p and 0 < s < q): return False
    h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    # Standard verification: v1 = (y^r * r^s) mod p, v2 = g^h mod p
    v1 = (pow(y, r, p) * pow(r, s, p)) % p; v2 = pow(g, h_int, p)
    return v1 == v2

# --- Other Utilities ---
def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')
def pause(): input("\nPress Enter to continue...")


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 1: COMPLETE 3-ALGORITHM SYSTEMS (UNCHANGED from Script 1)
# Systems 1-12 - PLACEHOLDERS - PASTE YOUR FULL CODE HERE
#
# ═══════════════════════════════════════════════════════════════════════════

# --- Placeholder functions for Systems 1-12 ---
# PASTE FULL CODE for menu_email_system() here
def menu_email_system(): print("\n[Placeholder for System 1: DES+RSA+SHA256]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_banking_system() here
def menu_banking_system(): print("\n[Placeholder for System 2: AES+ElGamal(Enc)+SHA512]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_cloud_system() here
def menu_cloud_system(): print("\n[Placeholder for System 3: Rabin+RSA+MD5]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_legacy_banking() here
def menu_legacy_banking(): print("\n[Placeholder for System 4: 3DES+ElGamal(Enc)+SHA1]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_healthcare() here
def menu_healthcare(): print("\n[Placeholder for System 5: AES+RSA+SHA256]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_document_management() here
def menu_document_management(): print("\n[Placeholder for System 6: DES+ElGamal(Enc)+MD5]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_messaging() here
def menu_messaging(): print("\n[Placeholder for System 7: AES+ElGamal(Enc)+MD5]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_file_transfer() here
def menu_file_transfer(): print("\n[Placeholder for System 8: DES+RSA+SHA512]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_digital_library() here
def menu_digital_library(): print("\n[Placeholder for System 9: Rabin+ElGamal(Enc)+SHA256]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_secure_chat() here
def menu_secure_chat(): print("\n[Placeholder for System 10: AES+RSA+SHA512]\nPASTE YOUR CODE HERE"); pause()
# PASTE FULL CODE for menu_voting_system() here (or use placeholder from V4)
def menu_voting_system():
    if HAS_PAILLIER: print("\n[Placeholder for System 11: Paillier+ElGamal(Sig)+SHA256 - E-Voting]\nPASTE YOUR CODE HERE"); pause()
    else: print("\nSystem 11: E-Voting requires 'phe' library."); pause()
# PASTE FULL CODE for menu_hill_hybrid() here (or use placeholder from V4)
def menu_hill_hybrid():
    if HAS_NUMPY: print("\n[Placeholder for System 12: Hill+RSA+SHA256]\nPASTE YOUR CODE HERE"); pause()
    else: print("\nSystem 12: Hill Hybrid requires 'numpy'."); pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 2: INTERACTIVE CLASSICAL CIPHERS
# Systems 13-20 (Integrated from ComprehensiveSecurityToolkit)
#
# ═══════════════════════════════════════════════════════════════════════════

# --- Implementations for Classical Ciphers ---
# (Using implementations from ComprehensiveSecurityToolkit script)
toolkit_classical = CryptographyToolkit() # Use the class containing implementations

# --- Interactive Menus for Each Classical Cipher ---

def menu_additive_cipher(): # System 13
    """Interactive Additive (Caesar) Cipher"""
    clear_screen(); print("🔤 SYSTEM 13: ADDITIVE CIPHER (CAESAR)"); print("="*40)
    message = input("Enter message: ").upper()
    try: key = int(input("Enter shift key (0-25): ")) % 26
    except ValueError: print("Invalid key."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.additive_cipher(message, key); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.additive_cipher(encrypted, key, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    # Log operation if needed
    pause()

def menu_multiplicative_cipher(): # System 14
    """Interactive Multiplicative Cipher"""
    clear_screen(); print("🔢 SYSTEM 14: MULTIPLICATIVE CIPHER"); print("="*40)
    message = input("Enter message: ").upper()
    print("Valid keys (coprime to 26): 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25")
    try: key = int(input("Enter key: "))
    except ValueError: print("Invalid key."); pause(); return

    if gcd(key, 26) != 1: print("❌ Invalid key! Must be coprime to 26."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.multiplicative_cipher(message, key); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.multiplicative_cipher(encrypted, key, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    print(f"Modular Inverse of {key}: {mod_inverse(key, 26)}")
    pause()

def menu_affine_cipher(): # System 15
    """Interactive Affine Cipher"""
    clear_screen(); print("🔡 SYSTEM 15: AFFINE CIPHER"); print("="*30)
    message = input("Enter message: ").upper()
    try:
        a = int(input("Enter 'a' (must be coprime to 26): "))
        b = int(input("Enter 'b' (0-25): ")) % 26
    except ValueError: print("Invalid key(s)."); pause(); return

    if gcd(a, 26) != 1: print("❌ Invalid 'a'! Must be coprime to 26."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.affine_cipher(message, a, b); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.affine_cipher(encrypted, a, b, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    print(f"Params: a={a}, b={b}, a_inv={mod_inverse(a, 26)}")
    pause()

def menu_vigenere_cipher(): # System 16
    """Interactive Vigenere Cipher"""
    clear_screen(); print("🔠 SYSTEM 16: VIGENERE CIPHER"); print("="*30)
    message = input("Enter message: ").upper()
    key = input("Enter keyword: ").upper()
    if not key.isalpha(): print("Invalid key."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.vigenere_cipher(message, key); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.vigenere_cipher(encrypted, key, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nKey:       {key}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    pause()

def menu_autokey_cipher(): # System 17
    """Interactive Autokey Cipher"""
    clear_screen(); print("🔑 SYSTEM 17: AUTOKEY CIPHER"); print("="*30)
    message = input("Enter message: ").upper()
    key = input("Enter keyword: ").upper()
    if not key.isalpha(): print("Invalid key."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.autokey_cipher(message, key); encrypt_time = time.time() - start_time
    # Decryption needs the encrypted text to generate its key stream correctly
    start_time = time.time(); decrypted = toolkit_classical.autokey_cipher(encrypted, key, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nKey:       {key}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    pause()

def menu_playfair_cipher(): # System 18
    """Interactive Playfair Cipher"""
    clear_screen(); print("▦ SYSTEM 18: PLAYFAIR CIPHER"); print("="*30)
    message = input("Enter message: ").upper()
    key = input("Enter keyword: ").upper()
    if not key.isalpha(): print("Invalid key."); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.playfair_cipher(message, key); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.playfair_cipher(encrypted, key, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nKey:       {key}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    pause()

def menu_hill_cipher(): # System 19
    """Interactive Hill Cipher (2x2)"""
    if not HAS_NUMPY: print("\nHill Cipher requires NumPy."); pause(); return
    clear_screen(); print("🔢 SYSTEM 19: HILL CIPHER (2x2)"); print("="*30)
    message = input("Enter message: ").upper()
    print("Enter 2x2 key matrix (e.g., '3 3 2 7'):")
    key_str = input("Key values: ").split()
    try:
        key_vals = [int(v) for v in key_str]
        if len(key_vals) != 4: raise ValueError("Need 4 values")
        key_matrix = [key_vals[0:2], key_vals[2:4]]
        # Check invertibility
        det = (key_matrix[0][0] * key_matrix[1][1] - key_matrix[0][1] * key_matrix[1][0]) % 26
        if mod_inverse(det, 26) is None: raise ValueError(f"Matrix determinant {det} not invertible mod 26")
    except ValueError as e: print(f"Invalid key: {e}"); pause(); return

    start_time = time.time(); encrypted = toolkit_classical.hill_cipher_2x2(message, key_matrix); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = toolkit_classical.hill_cipher_2x2(encrypted, key_matrix, decrypt=True); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nKey Matrix:{key_matrix}\nEncrypted: {encrypted}\nDecrypted: {decrypted}")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    pause()

def menu_transposition_cipher(): # System 20
    """Interactive Transposition Cipher (Columnar)"""
    clear_screen(); print("🔄 SYSTEM 20: TRANSPOSITION CIPHER (COLUMNAR)"); print("="*50)
    message = input("Enter message: ").upper().replace(" ", "")
    key = input("Enter keyword: ").upper()
    if not key.isalpha() or len(key) == 0: print("Invalid key."); pause(); return

    # Simple Columnar Transposition implementation
    def transpose_encrypt(msg, keyword):
        key_order = sorted([(char, i) for i, char in enumerate(keyword)])
        num_cols = len(keyword)
        num_rows = math.ceil(len(msg) / num_cols)
        grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        msg_idx = 0
        for r in range(num_rows):
            for c in range(num_cols):
                if msg_idx < len(msg): grid[r][c] = msg[msg_idx]; msg_idx += 1
                else: grid[r][c] = 'X' # Padding

        ciphertext = ""
        for _, c in key_order:
            for r in range(num_rows): ciphertext += grid[r][c]
        return ciphertext

    def transpose_decrypt(cipher, keyword):
        key_order = sorted([(char, i) for i, char in enumerate(keyword)])
        num_cols = len(keyword)
        num_rows = math.ceil(len(cipher) / num_cols)
        if len(cipher) != num_cols * num_rows: return "Ciphertext length error" # Basic check

        # Calculate number of full/short columns
        num_full_cols = len(cipher) % num_cols
        if num_full_cols == 0: num_full_cols = num_cols

        grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        cipher_idx = 0
        col_lengths = [num_rows] * num_cols
        # Adjust for shorter columns at the end if padding wasn't exact multiple
        num_short_cols = num_cols - (len(cipher) % num_cols) if len(cipher) % num_cols != 0 else 0
        
        simulated_msg_len = len(cipher) # Assume padding fills grid
        
        # Determine column lengths based on key order and potential padding removal needs
        # This part is tricky without knowing original length or padding rules exactly.
        # Assuming simple padding fill for decryption grid reconstruction:
        
        col_ptr = 0
        for _, orig_col_idx in key_order:
            # How many chars belong in this column?
            col_len = num_rows if orig_col_idx < (num_cols - num_short_cols) else num_rows -1 # Incorrect if padding added more 'X's

            # A simpler way assuming full grid padding:
            col_len = num_rows
            
            for r in range(col_len):
                 if cipher_idx < len(cipher):
                      grid[r][orig_col_idx] = cipher[cipher_idx]
                      cipher_idx += 1


        plaintext = ""
        for r in range(num_rows):
             for c in range(num_cols):
                 plaintext += grid[r][c]
        # Crude padding removal - might remove intended 'X's
        return plaintext.rstrip('X')


    start_time = time.time(); encrypted = transpose_encrypt(message, key); encrypt_time = time.time() - start_time
    start_time = time.time(); decrypted = transpose_decrypt(encrypted, key); decrypt_time = time.time() - start_time

    print("\n📊 RESULTS:")
    print(f"Original:  {message}\nKey:       {key}\nEncrypted: {encrypted}\nDecrypted: {decrypted} (Note: Decryption/Padding removal is basic)")
    print(f"Times (Encrypt/Decrypt): {encrypt_time:.6f}s / {decrypt_time:.6f}s")
    pause()

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 3: EXAM-SPECIFIC SCENARIOS & NEW COMBINATIONS
# Systems 21+ (Includes 19-22 from V4, renumbered to 21-24, plus new ones)
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 21: Client-Server Payment Gateway (Paillier + RSA Sign + SHA-256) ---
# (Code for PaymentGatewaySystem Class and menu_payment_gateway() from V4 goes here)
# Renumber System 19 -> 21
if HAS_PAILLIER and HAS_CRYPTO:
    # --- PASTE PaymentGatewaySystem CLASS CODE HERE ---
    class PaymentGatewaySystem:
        """Simulates Seller-Payment Gateway transactions using Paillier, RSA, SHA-256"""
        # (Code from V4)
        def __init__(self, paillier_key_size=1024, rsa_key_size=2048):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Payment Gateway (Paillier {paillier_key_size}, RSA {rsa_key_size})...")
            start = time.time(); self.paillier_public_key, self.paillier_private_key = paillier.generate_paillier_keypair(n_length=paillier_key_size); self.performance.record('Paillier_KeyGen', time.time() - start); print("  ✓ Paillier keys generated.")
            start = time.time(); self.gateway_rsa_key = RSA.generate(rsa_key_size); self.gateway_rsa_public_key = self.gateway_rsa_key.publickey(); self.performance.record('RSA_KeyGen_Gateway', time.time() - start); print("  ✓ Gateway RSA keys generated.")
            self.sellers = {}; self.transaction_log = []
            print("Payment Gateway Initialized.")
        def register_seller(self, seller_name, rsa_key_size=2048):
            if seller_name in self.sellers: print(f"Seller '{seller_name}' already registered."); return False
            start = time.time(); seller_rsa_key = RSA.generate(rsa_key_size); seller_rsa_public_key = seller_rsa_key.publickey(); self.performance.record('RSA_KeyGen_Seller', time.time() - start)
            self.sellers[seller_name] = {'transactions': [], 'rsa_private_key': seller_rsa_key, 'rsa_public_key': seller_rsa_public_key, 'total_encrypted': self.paillier_public_key.encrypt(0), 'total_decrypted': 0.0}
            print(f"✓ Seller '{seller_name}' registered."); return True
        def seller_submit_transaction(self, seller_name, amount):
            if seller_name not in self.sellers: print(f"Error: Seller '{seller_name}' not registered."); return None
            if not isinstance(amount, (int, float)) or amount <= 0: print("Error: Invalid amount."); return None
            seller_data = self.sellers[seller_name]; tx_id = f"{seller_name}_{len(seller_data['transactions']) + 1}"
            start = time.time(); encrypted_amount = self.paillier_public_key.encrypt(amount); enc_time = time.time() - start; self.performance.record('Paillier_Encrypt', enc_time, data_size=sys.getsizeof(amount))
            tx_details = {'id': tx_id, 'amount': amount, 'enc_amount_val': encrypted_amount.ciphertext(), 'enc_amount_exp': encrypted_amount.exponent, 'dec_amount': None}
            seller_data['transactions'].append(tx_details)
            start = time.time(); seller_data['total_encrypted'] = seller_data['total_encrypted'] + encrypted_amount; add_time = time.time() - start; self.performance.record('Paillier_Add', add_time)
            print(f"  Seller '{seller_name}' submitted {tx_id}: Amount={amount} (Enc: {enc_time:.6f}s, Add: {add_time:.6f}s)"); return seller_name, tx_id, encrypted_amount
        def gateway_process_transactions(self, submissions):
            print("\n--- Gateway Processing ---");
            if not submissions: print("No transactions submitted."); return
            processed_sellers = set(s[0] for s in submissions) # Process totals only for sellers who submitted
            for seller_name in processed_sellers:
                if seller_name not in self.sellers: continue
                seller_data = self.sellers[seller_name]
                print(f"Processing total for Seller '{seller_name}'...")
                start = time.time(); total_decrypted = self.paillier_private_key.decrypt(seller_data['total_encrypted']); dec_time = time.time() - start; self.performance.record('Paillier_Decrypt_Total', dec_time); seller_data['total_decrypted'] = total_decrypted
                print(f"  Decrypted Total: {total_decrypted} (Took: {dec_time:.6f}s)")
                for tx in seller_data['transactions']:
                    if tx['dec_amount'] is None:
                         enc_obj = paillier.EncryptedNumber(self.paillier_public_key, tx['enc_amount_val'], tx['enc_amount_exp'])
                         start_ind = time.time(); tx['dec_amount'] = self.paillier_private_key.decrypt(enc_obj); dec_ind_time = time.time() - start_ind; self.performance.record('Paillier_Decrypt_Individual', dec_ind_time)
        def generate_transaction_summary(self, seller_name):
            if seller_name not in self.sellers: return "Seller not found."
            seller_data = self.sellers[seller_name]; summary = f"--- Summary: {seller_name} ({datetime.now().isoformat()}) ---\nTransactions:\n"
            if not seller_data['transactions']: summary += "  No transactions.\n"
            else:
                 for tx in seller_data['transactions']: summary += f"  ID: {tx['id']}, Amount: {tx['amount']}, Enc: {str(tx['enc_amount_val'])[:20]}..., Dec: {tx['dec_amount']}\n"
            total_enc_str = str(seller_data['total_encrypted'].ciphertext()); summary += f"\nTotal Enc: {total_enc_str[:30]}...\nTotal Dec: {seller_data['total_decrypted']}\nSig Status: [Not Signed Yet]\nVerification: [N/A]\n-------------------------------------------\n"
            return summary
        def seller_sign_summary(self, seller_name, summary_string):
            if seller_name not in self.sellers: print("Error: Seller not found."); return None, None
            seller_data = self.sellers[seller_name]
            start = time.time(); summary_bytes = summary_string.encode('utf-8'); hash_obj = SHA256.new(summary_bytes); hash_time = time.time() - start; self.performance.record('SHA256_Hash', hash_time, data_size=len(summary_bytes))
            start = time.time()
            try: signature = pkcs1_15.new(seller_data['rsa_private_key']).sign(hash_obj); sign_time = time.time() - start; self.performance.record('RSA_Sign', sign_time); print(f"  Seller '{seller_name}' signed summary (SHA256: {hash_time:.6f}s, RSA Sign: {sign_time:.6f}s)"); return signature, hash_obj
            except Exception as e: print(f"Error signing summary for {seller_name}: {e}"); return None, None
        def gateway_verify_signature(self, seller_name, summary_string, signature, hash_obj):
             if seller_name not in self.sellers: print("Seller not found."); return False
             if not signature or not hash_obj: print("Missing signature/hash."); return False
             seller_public_key = self.sellers[seller_name]['rsa_public_key']
             start = time.time()
             try: pkcs1_15.new(seller_public_key).verify(hash_obj, signature); verify_time = time.time() - start; self.performance.record('RSA_Verify', verify_time); print(f"  Gateway verified '{seller_name}': VALID (Took: {verify_time:.6f}s)"); return True
             except (ValueError, TypeError): verify_time = time.time() - start; self.performance.record('RSA_Verify', verify_time); print(f"  Gateway verified '{seller_name}': INVALID (Took: {verify_time:.6f}s)"); return False
             except Exception as e: print(f"Verification error for {seller_name}: {e}"); return False

    # --- PASTE menu_payment_gateway() FUNCTION CODE HERE ---
    def menu_payment_gateway(): # System 21
        # (Code from V4, adjusted for System 21 numbering)
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 21 requires 'phe' and 'pycryptodome'."); pause(); return
        system = PaymentGatewaySystem(); submitted_transactions = []
        while True:
            print(f"\n{'='*70}\n  🏦 SYSTEM 21: SELLER-PAYMENT GATEWAY (Paillier + RSA Sign + SHA-256)\n{'='*70}")
            print("  1. Register Seller | 2. Submit Transaction | 3. Gateway: Process Transactions | 4. Generate & Sign Summary | 5. Verify Signed Summary | 6. View All Summaries | 7. Performance | 8. Compare Algs | 9. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            # Condensed menu logic
            if choice == '1': s_name = input("Seller name: "); system.register_seller(s_name)
            elif choice == '2':
                s_name = input("Seller name: ");
                if s_name not in system.sellers: print("Seller not registered."); continue
                try: amount = float(input("Amount: ")); submission = system.seller_submit_transaction(s_name, amount);
                    if submission: submitted_transactions.append(submission)
                except ValueError as e: print(f"Invalid amount: {e}")
            elif choice == '3': system.gateway_process_transactions(submitted_transactions); submitted_transactions = []
            elif choice == '4':
                s_name = input("Seller name: ");
                if s_name not in system.sellers: print("Seller not registered."); continue
                summary = system.generate_transaction_summary(s_name); print(f"\nGenerated:\n{summary}")
                signature, hash_obj = system.seller_sign_summary(s_name, summary)
                if signature: system.sellers[s_name]['last_summary'] = summary; system.sellers[s_name]['last_signature'] = signature; system.sellers[s_name]['last_hash_obj'] = hash_obj; print("Summary signed.")
                else: print("Signing failed.")
            elif choice == '5':
                 s_name = input("Seller name: ");
                 if s_name not in system.sellers: print("Seller not registered."); continue
                 if 'last_summary' not in system.sellers[s_name]: print("No signed summary found."); continue
                 summary, signature, hash_obj = system.sellers[s_name]['last_summary'], system.sellers[s_name]['last_signature'], system.sellers[s_name]['last_hash_obj']
                 print(f"\nVerifying:\n{summary}"); is_valid = system.gateway_verify_signature(s_name, summary, signature, hash_obj)
                 final_summary = summary.replace("[Not Signed Yet]", signature.hex()[:40]+"...").replace("[N/A]", "VALID" if is_valid else "INVALID"); print(f"\nFinal Status:\n{final_summary}")
            elif choice == '6': print("\n--- All Summaries ---"); [print(system.generate_transaction_summary(s)) for s in system.sellers] if system.sellers else print("No sellers.")
            elif choice == '7': system.performance.print_graph("Payment Gateway Performance")
            elif choice == '8': system.performance.compare_algorithms(['Paillier_Encrypt', 'RSA_Sign', 'SHA256_Hash'])
            elif choice == '9': break
            else: print("Invalid choice.")
            if choice not in ['7', '8', '9']: pause()

else:
    def menu_payment_gateway(): print("\nSystem 21: Requires 'phe' and 'pycryptodome'."); pause()


# --- System 22: Secure Aggregation (Paillier + ElGamal Sign + SHA-512) ---
# (Code for SecureAggregationPaillierElGamal Class and menu function from V4 goes here)
# Renumber System 20 -> 22
if HAS_PAILLIER and HAS_CRYPTO:
    # --- PASTE SecureAggregationPaillierElGamal CLASS CODE HERE ---
    class SecureAggregationPaillierElGamal:
        """Paillier aggregation with ElGamal signatures"""
        # (Code from V4)
        def __init__(self, paillier_bits=1024, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Aggregation (Paillier {paillier_bits}, ElGamal {elgamal_bits})...")
            start = time.time(); self.paillier_pub, self.paillier_priv = paillier.generate_paillier_keypair(n_length=paillier_bits); self.performance.record('Paillier_KeyGen', time.time()-start); print("  ✓ Paillier keys.")
            start = time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen', time.time()-start); print("  ✓ ElGamal keys.")
            self.contributions = []; self.encrypted_sum = self.paillier_pub.encrypt(0)
        def submit_value(self, participant_id, value):
            print(f"\nSubmitting {value} from {participant_id}...")
            start=time.time(); enc_value = self.paillier_pub.encrypt(value); self.performance.record('Paillier_Encrypt', time.time()-start)
            data_to_sign = f"{participant_id}:{enc_value.ciphertext()}:{enc_value.exponent}".encode('utf-8')
            start=time.time(); hash_obj = SHA512.new(data_to_sign); hash_val = hash_obj.digest(); self.performance.record('SHA512_Hash', time.time()-start)
            start=time.time(); signature = elgamal_sign(hash_val, self.elgamal_keys); self.performance.record('ElGamal_Sign', time.time()-start)
            submission = {'id': participant_id, 'enc_value': enc_value, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            self.contributions.append(submission)
            start=time.time(); self.encrypted_sum += enc_value; self.performance.record('Paillier_Add', time.time()-start)
            print("  ✓ Submitted & added."); return submission
        def verify_and_decrypt_sum(self):
            print("\n--- Aggregator Processing ---"); verified_count = 0
            elg_pub_key = {k:v for k,v in self.elgamal_keys.items() if k != 'x'}
            for sub in self.contributions:
                data_signed = f"{sub['id']}:{sub['enc_value'].ciphertext()}:{sub['enc_value'].exponent}".encode('utf-8')
                hash_recomputed = SHA512.new(data_signed).digest()
                start = time.time(); is_valid = elgamal_verify(hash_recomputed, sub['signature'], elg_pub_key); self.performance.record('ElGamal_Verify', time.time()-start)
                if is_valid: verified_count += 1
                else: print(f"  ⚠ Sig verify FAILED for {sub['id']}!")
            print(f"\nVerified {verified_count}/{len(self.contributions)} signatures.")
            start = time.time(); final_sum = self.paillier_priv.decrypt(self.encrypted_sum); self.performance.record('Paillier_Decrypt_Total', time.time()-start)
            print(f"\nDecrypted Final Sum: {final_sum}"); return final_sum

    # --- PASTE menu_secure_aggregation_paillier_elgamal() FUNCTION CODE HERE ---
    def menu_secure_aggregation_paillier_elgamal(): # System 22
        # (Code from V4, adjusted for System 22 numbering)
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 22 requires 'phe' and 'pycryptodome'."); pause(); return
        system = SecureAggregationPaillierElGamal(); participant_count = 0
        while True:
            print(f"\n{'='*70}\n  ∑ SYSTEM 22: SECURE AGGREGATION (Paillier + ElGamal Sign + SHA-512)\n{'='*70}")
            print("  1. Submit Value | 2. Verify & Decrypt Sum | 3. View Contributions | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1':
                participant_count += 1; pid = f"P{participant_count}"
                try: value = int(input(f"Value for {pid}: ")); system.submit_value(pid, value)
                except ValueError: print("Invalid value.")
            elif choice == '2': system.verify_and_decrypt_sum()
            elif choice == '3':
                print("\n--- Contributions ---");
                [print(f"  ID: {sub['id']}, Enc: {str(sub['enc_value'].ciphertext())[:20]}..., Hash: {sub['hash_hex'][:16]}..., Sig: {sub['signature']}") for sub in system.contributions] if system.contributions else print("None.")
            elif choice == '4': system.performance.print_graph("Secure Aggregation (P+ElG) Performance")
            elif choice == '5': system.performance.compare_algorithms(['Paillier_Encrypt', 'ElGamal_Sign', 'SHA512_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_secure_aggregation_paillier_elgamal(): print("\nSystem 22: Requires 'phe' and 'pycryptodome'."); pause()


# --- System 23: Homomorphic Product Demo (ElGamal Enc + RSA Sign + SHA256) ---
# (Code for HomomorphicProductElGamal Class and menu function from V4 goes here)
# Renumber System 21 -> 23
if HAS_CRYPTO:
    # --- PASTE HomomorphicProductElGamal CLASS CODE HERE ---
    class HomomorphicProductElGamal:
        """ElGamal multiplicative homomorphism with RSA signatures"""
        # (Code from V4)
        def __init__(self, elgamal_bits=1024, rsa_bits=2048):
             self.performance = PerformanceTracker()
             print(f"\nInitializing Homomorphic Product (ElGamal {elgamal_bits}, RSA {rsa_bits})...")
             start = time.time(); self.elgamal_enc_key = ElGamal.generate(elgamal_bits, get_random_bytes); self.performance.record('ElGamal_KeyGen_Enc', time.time()-start); print("  ✓ ElGamal keys.")
             start = time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Sign', time.time()-start); print("  ✓ RSA keys.")
             self.encrypted_factors = []; self.encrypted_product_c1 = 1; self.encrypted_product_c2 = 1
        def submit_factor(self, factor_id, factor_value):
            if not isinstance(factor_value, int) or factor_value <= 0: print("Factor must be positive integer."); return None
            print(f"\nSubmitting factor {factor_value} from {factor_id}...")
            p=self.elgamal_enc_key.p; g=self.elgamal_enc_key.g; y=self.elgamal_enc_key.y
            start=time.time(); k = number.getRandomRange(1, p-1); c1 = pow(g, k, p); c2 = (factor_value * pow(y, k, p)) % p; enc_value = (c1, c2); self.performance.record('ElGamal_Encrypt', time.time()-start)
            data_to_sign = f"{factor_id}:{enc_value[0]}:{enc_value[1]}".encode('utf-8')
            start=time.time(); hash_obj = SHA256.new(data_to_sign); self.performance.record('SHA256_Hash', time.time()-start)
            start=time.time(); signature = pkcs1_15.new(self.rsa_key).sign(hash_obj); self.performance.record('RSA_Sign', time.time()-start)
            submission = {'id': factor_id, 'enc_value': enc_value, 'signature': signature, 'hash_hex': hash_obj.hexdigest(), 'original_value': factor_value} # Store original for demo check
            self.encrypted_factors.append(submission)
            start=time.time(); self.encrypted_product_c1 = (self.encrypted_product_c1 * enc_value[0]) % p; self.encrypted_product_c2 = (self.encrypted_product_c2 * enc_value[1]) % p; self.performance.record('ElGamal_Multiply', time.time()-start)
            print("  ✓ Submitted & multiplied."); return submission
        def verify_and_decrypt_product(self):
             print("\n--- Aggregator Processing ---"); verified_count = 0
             for sub in self.encrypted_factors:
                 data_signed = f"{sub['id']}:{sub['enc_value'][0]}:{sub['enc_value'][1]}".encode('utf-8'); hash_obj_recomputed = SHA256.new(data_signed)
                 start=time.time();
                 try: pkcs1_15.new(self.rsa_pub_key).verify(hash_obj_recomputed, sub['signature']); is_valid = True
                 except (ValueError, TypeError): is_valid = False
                 self.performance.record('RSA_Verify', time.time()-start)
                 if is_valid: verified_count += 1
                 else: print(f"  ⚠ RSA Sig verify FAILED for {sub['id']}!")
             print(f"\nVerified {verified_count}/{len(self.encrypted_factors)} signatures.")
             p=self.elgamal_enc_key.p; x=self.elgamal_enc_key.x; c1_prod=self.encrypted_product_c1; c2_prod=self.encrypted_product_c2
             start=time.time(); s = pow(c1_prod, x, p); s_inv = number.inverse(s, p); final_product = (c2_prod * s_inv) % p; self.performance.record('ElGamal_Decrypt_Product', time.time()-start)
             print(f"\nDecrypted Final Product: {final_product}")
             original_product = 1; [original_product := (original_product * sub['original_value']) % p for sub in self.encrypted_factors]; print(f"Verification: Product of originals (mod p) = {original_product} | Match: {original_product == final_product}"); return final_product

    # --- PASTE menu_homomorphic_product_elgamal() FUNCTION CODE HERE ---
    def menu_homomorphic_product_elgamal(): # System 23
        # (Code from V4, adjusted for System 23 numbering)
        if not HAS_CRYPTO: print("\nSystem 23 requires 'pycryptodome'."); pause(); return
        system = HomomorphicProductElGamal(); participant_count = 0
        while True:
            print(f"\n{'='*70}\n  ∏ SYSTEM 23: HOMOMORPHIC PRODUCT (ElGamal Enc + RSA Sign + SHA256)\n{'='*70}")
            print("  1. Submit Factor | 2. Verify & Decrypt Product | 3. View Factors | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1':
                participant_count += 1; pid = f"F{participant_count}"
                try: value = int(input(f"Factor for {pid}: ")); system.submit_factor(pid, value)
                except ValueError as e: print(f"Invalid factor: {e}")
            elif choice == '2': system.verify_and_decrypt_product()
            elif choice == '3':
                print("\n--- Factors ---");
                [print(f"  ID: {sub['id']}, Enc: ({str(sub['enc_value'][0])[:10]}..., {str(sub['enc_value'][1])[:10]}...), Hash: {sub['hash_hex'][:16]}..., Sig: {sub['signature'].hex()[:16]}...") for sub in system.encrypted_factors] if system.encrypted_factors else print("None.")
            elif choice == '4': system.performance.print_graph("Homomorphic Product Performance")
            elif choice == '5': system.performance.compare_algorithms(['ElGamal_Encrypt', 'RSA_Sign', 'SHA256_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_homomorphic_product_elgamal(): print("\nSystem 23: Requires 'pycryptodome'."); pause()


# --- System 24: Secure Aggregation (Paillier + RSA Sign + SHA-512) ---
# (Code for SecureAggregationPaillierRSA Class and menu function from V4 goes here)
# Renumber System 22 -> 24
if HAS_PAILLIER and HAS_CRYPTO:
    # --- PASTE SecureAggregationPaillierRSA CLASS CODE HERE ---
    class SecureAggregationPaillierRSA:
        """Paillier aggregation with RSA signatures"""
        # (Code from V4)
        def __init__(self, paillier_bits=1024, rsa_bits=2048):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Aggregation (Paillier {paillier_bits}, RSA {rsa_bits})...")
            start=time.time(); self.paillier_pub, self.paillier_priv = paillier.generate_paillier_keypair(n_length=paillier_bits); self.performance.record('Paillier_KeyGen', time.time()-start); print("  ✓ Paillier keys.")
            start=time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Sign', time.time()-start); print("  ✓ RSA keys.")
            self.contributions = []; self.encrypted_sum = self.paillier_pub.encrypt(0)
        def submit_value(self, participant_id, value):
            print(f"\nSubmitting {value} from {participant_id}...")
            start=time.time(); enc_value = self.paillier_pub.encrypt(value); self.performance.record('Paillier_Encrypt', time.time()-start)
            data_to_sign = f"{participant_id}:{enc_value.ciphertext()}:{enc_value.exponent}".encode('utf-8')
            start=time.time(); hash_obj = SHA512.new(data_to_sign); self.performance.record('SHA512_Hash', time.time()-start)
            start=time.time(); signature = pkcs1_15.new(self.rsa_key).sign(hash_obj); self.performance.record('RSA_Sign', time.time()-start)
            submission = {'id': participant_id, 'enc_value': enc_value, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            self.contributions.append(submission)
            start=time.time(); self.encrypted_sum += enc_value; self.performance.record('Paillier_Add', time.time()-start)
            print("  ✓ Submitted & added."); return submission
        def verify_and_decrypt_sum(self):
            print("\n--- Aggregator Processing ---"); verified_count = 0
            for sub in self.contributions:
                data_signed = f"{sub['id']}:{sub['enc_value'].ciphertext()}:{sub['enc_value'].exponent}".encode('utf-8'); hash_obj_recomputed = SHA512.new(data_signed)
                start=time.time();
                try: pkcs1_15.new(self.rsa_pub_key).verify(hash_obj_recomputed, sub['signature']); is_valid = True
                except (ValueError, TypeError): is_valid = False
                self.performance.record('RSA_Verify', time.time()-start)
                if is_valid: verified_count += 1
                else: print(f"  ⚠ RSA Sig verify FAILED for {sub['id']}!")
            print(f"\nVerified {verified_count}/{len(self.contributions)} signatures.")
            start = time.time(); final_sum = self.paillier_priv.decrypt(self.encrypted_sum); self.performance.record('Paillier_Decrypt_Total', time.time()-start)
            print(f"\nDecrypted Final Sum: {final_sum}"); return final_sum

    # --- PASTE menu_secure_aggregation_paillier_rsa() FUNCTION CODE HERE ---
    def menu_secure_aggregation_paillier_rsa(): # System 24
        # (Code from V4, adjusted for System 24 numbering)
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 24 requires 'phe' and 'pycryptodome'."); pause(); return
        system = SecureAggregationPaillierRSA(); participant_count = 0
        while True:
            print(f"\n{'='*70}\n  ∑ SYSTEM 24: SECURE AGGREGATION (Paillier + RSA Sign + SHA-512)\n{'='*70}")
            print("  1. Submit Value | 2. Verify & Decrypt Sum | 3. View Contributions | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1':
                participant_count += 1; pid = f"P{participant_count}"
                try: value = int(input(f"Value for {pid}: ")); system.submit_value(pid, value)
                except ValueError: print("Invalid value.")
            elif choice == '2': system.verify_and_decrypt_sum()
            elif choice == '3':
                 print("\n--- Contributions ---");
                 [print(f"  ID: {sub['id']}, Enc: {str(sub['enc_value'].ciphertext())[:20]}..., Hash: {sub['hash_hex'][:16]}..., Sig: {sub['signature'].hex()[:16]}...") for sub in system.contributions] if system.contributions else print("None.")
            elif choice == '4': system.performance.print_graph("Secure Aggregation (P+RSA) Performance")
            elif choice == '5': system.performance.compare_algorithms(['Paillier_Encrypt', 'RSA_Sign', 'SHA512_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_secure_aggregation_paillier_rsa(): print("\nSystem 24: Requires 'phe' and 'pycryptodome'."); pause()


# --- System 25: AES + ElGamal Sign + SHA256 (Secure Data Transmission) ---
if HAS_CRYPTO:
    class SecureTransmissionAESElGamal:
        def __init__(self, aes_key_size=32, elgamal_bits=1024): # AES-256
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Transmission (AES-{aes_key_size*8}, ElGamal {elgamal_bits})...")
            start=time.time(); self.aes_key = get_random_bytes(aes_key_size); self.performance.record('AES_KeyGen', time.time()-start) # Symmetric key
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen', time.time()-start); print("  ✓ ElGamal signature keys generated.")
            self.transmissions = [] # Log of {'id', 'enc_data', 'nonce', 'tag', 'signature', 'hash_hex'}

        def send_data(self, sender_id, data):
            """Sender encrypts data with AES, signs hash with ElGamal"""
            print(f"\nSender {sender_id} sending data...")
            data_bytes = data.encode('utf-8')

            # Encrypt with AES (GCM mode recommended for authenticated encryption)
            start = time.time()
            cipher_aes = AES.new(self.aes_key, AES.MODE_GCM)
            nonce = cipher_aes.nonce
            enc_data, tag = cipher_aes.encrypt_and_digest(data_bytes)
            self.performance.record('AES_Encrypt_GCM', time.time()-start, len(data_bytes))

            # Hash encrypted data + nonce + tag for signing
            data_to_sign = enc_data + nonce + tag
            start = time.time()
            hash_obj = SHA256.new(data_to_sign)
            hash_val = hash_obj.digest()
            self.performance.record('SHA256_Hash', time.time()-start)

            # Sign hash with ElGamal
            start = time.time()
            signature = elgamal_sign(hash_val, self.elgamal_keys) # Use private key 'x'
            self.performance.record('ElGamal_Sign', time.time()-start)

            transmission = {
                'id': len(self.transmissions) + 1,
                'sender': sender_id,
                'enc_data': enc_data,
                'nonce': nonce,
                'tag': tag,
                'signature': signature,
                'hash_hex': hash_obj.hexdigest()
            }
            self.transmissions.append(transmission)
            print("  ✓ Data encrypted, signed, and logged for transmission.")
            return transmission # Simulate sending this bundle

        def receive_data(self, transmission):
            """Receiver verifies signature, then decrypts data"""
            print(f"\nReceiver processing transmission ID {transmission['id']} from {transmission['sender']}...")
            elg_pub_key = {k:v for k,v in self.elgamal_keys.items() if k != 'x'} # Public key for verification

            # Reconstruct data that was signed
            data_signed = transmission['enc_data'] + transmission['nonce'] + transmission['tag']
            hash_recomputed = SHA256.new(data_signed).digest()

            # Verify signature first
            start = time.time()
            is_valid_sig = elgamal_verify(hash_recomputed, transmission['signature'], elg_pub_key)
            self.performance.record('ElGamal_Verify', time.time()-start)

            if not is_valid_sig:
                print("  ❌ INVALID SIGNATURE! Data rejected.")
                return None

            print("  ✓ Signature VALID.")

            # Decrypt data using AES GCM
            start = time.time()
            try:
                cipher_aes = AES.new(self.aes_key, AES.MODE_GCM, nonce=transmission['nonce'])
                decrypted_data = cipher_aes.decrypt_and_verify(transmission['enc_data'], transmission['tag'])
                dec_time = time.time()-start
                self.performance.record('AES_Decrypt_GCM', dec_time)
                print(f"  ✓ Data decrypted successfully (Took: {dec_time:.6f}s)")
                return decrypted_data.decode('utf-8')
            except (ValueError, KeyError) as e:
                dec_time = time.time()-start
                self.performance.record('AES_Decrypt_GCM', dec_time) # Record even on failure
                print(f"  ❌ Decryption FAILED (Integrity check failed?): {e}")
                return None

    def menu_secure_transmission_aes_elgamal(): # System 25
        if not HAS_CRYPTO: print("\nSystem 25 requires 'pycryptodome'."); pause(); return

        system = SecureTransmissionAESElGamal()
        last_transmission = None

        while True:
            print(f"\n{'='*70}\n  🔒 SYSTEM 25: SECURE TRANSMISSION (AES-GCM + ElGamal Sign + SHA256)\n{'='*70}")
            print("  1. Send Data (Encrypt & Sign)")
            print("  2. Receive Data (Verify & Decrypt)")
            print("  3. View Last Transmission Details")
            print("  4. Performance Analysis")
            print("  5. Compare Algs (AES vs ElGamal vs SHA256)")
            print("  6. Back to Main Menu"); print("-"*70)
            choice = input("Choice: ").strip()

            if choice == '1':
                sender = input("Sender ID: ")
                data = input("Data to send: ")
                last_transmission = system.send_data(sender, data)
            elif choice == '2':
                if last_transmission:
                     decrypted = system.receive_data(last_transmission)
                     if decrypted is not None: print(f"\n  Decrypted Data: {decrypted}")
                else: print("No data has been sent yet in this session.")
            elif choice == '3':
                 if last_transmission:
                     print("\n--- Last Transmission ---")
                     print(f"  ID: {last_transmission['id']}, Sender: {last_transmission['sender']}")
                     print(f"  Enc Data: {last_transmission['enc_data'].hex()[:40]}...")
                     print(f"  Nonce: {last_transmission['nonce'].hex()}")
                     print(f"  Tag: {last_transmission['tag'].hex()}")
                     print(f"  Hash: {last_transmission['hash_hex'][:32]}...")
                     print(f"  Sig (r,s): {last_transmission['signature']}")
                 else: print("No transmission details available.")
            elif choice == '4': system.performance.print_graph("Secure Transmission Performance")
            elif choice == '5': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'ElGamal_Sign', 'SHA256_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_secure_transmission_aes_elgamal(): print("\nSystem 25: Requires 'pycryptodome'."); pause()


# --- System 26: Rabin + RSA Sign + SHA512 (Secure File Storage) ---
if HAS_CRYPTO:
    class SecureStorageRabinRSA:
        def __init__(self, rabin_bits=2048, rsa_bits=2048):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Storage (Rabin {rabin_bits}, RSA {rsa_bits})...")
            start=time.time(); self.rabin_keys = self._generate_rabin_keys(rabin_bits); self.performance.record('Rabin_KeyGen', time.time()-start); print("  ✓ Rabin keys generated.")
            start=time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Sign', time.time()-start); print("  ✓ RSA keys generated.")
            self.files = {} # filename: {'enc_content', 'signature', 'hash_hex'}

        def _generate_rabin_keys(self, bits):
             # (Using Blum integers p, q = 3 mod 4)
             while True: p = number.getPrime(bits // 2);
                 if p % 4 == 3: break
             while True: q = number.getPrime(bits // 2);
                 if q % 4 == 3 and q != p: break
             return {'n': p * q, 'p': p, 'q': q}

        def _rabin_encrypt(self, msg_bytes, n):
            # Add simple padding/redundancy for decryption disambiguation
            redundancy = b"FILEPAD" + len(msg_bytes).to_bytes(4, 'big') # 4 bytes for length
            full_msg = msg_bytes + redundancy
            m = number.bytes_to_long(full_msg)
            if m >= n: raise ValueError("Message too large for Rabin key")
            return pow(m, 2, n)

        def _rabin_decrypt(self, cipher_int, p, q, n):
            # (CRT method from V4 utilities - adapted)
            mp = pow(cipher_int, (p + 1) // 4, p); mq = pow(cipher_int, (q + 1) // 4, q)
            inv_p=number.inverse(p, q); inv_q=number.inverse(q, p)
            a=(q * inv_q) % n; b=(p * inv_p) % n
            roots = [(a*mp+b*mq)%n, (a*mp-b*mq)%n, (-a*mp+b*mq)%n, (-a*mp-b*mq)%n]
            for r in roots:
                try:
                     r_bytes = number.long_to_bytes(r)
                     # Check for padding "FILEPAD" + 4-byte length
                     if len(r_bytes) > 11 and r_bytes.endswith(b"FILEPAD"):
                         len_bytes_start = len(r_bytes) - 11
                         msg_len = int.from_bytes(r_bytes[len_bytes_start:len_bytes_start+4], 'big')
                         if len_bytes_start == msg_len: # Check if lengths match
                              return r_bytes[:msg_len]
                except Exception: continue
            return None # Decryption failed

        def store_file(self, filename, content):
            """Encrypt content with Rabin, sign hash with RSA"""
            print(f"\nStoring file '{filename}'...")
            content_bytes = content.encode('utf-8')

            # Encrypt with Rabin
            start=time.time()
            try: enc_content = self._rabin_encrypt(content_bytes, self.rabin_keys['n'])
            except ValueError as e: print(f"  Error encrypting: {e}"); return False
            self.performance.record('Rabin_Encrypt', time.time()-start, len(content_bytes))

            # Hash encrypted content (as int -> bytes) + filename with SHA-512
            data_to_sign = number.long_to_bytes(enc_content) + filename.encode('utf-8')
            start=time.time(); hash_obj = SHA512.new(data_to_sign); self.performance.record('SHA512_Hash', time.time()-start)

            # Sign hash with RSA
            start=time.time(); signature = pkcs1_15.new(self.rsa_key).sign(hash_obj); self.performance.record('RSA_Sign', time.time()-start)

            self.files[filename] = {
                'enc_content': enc_content, # Store as integer
                'signature': signature,
                'hash_hex': hash_obj.hexdigest()
            }
            print("  ✓ File encrypted, signed, and stored.")
            return True

        def retrieve_file(self, filename):
            """Verify signature, then decrypt content"""
            if filename not in self.files: print(f"File '{filename}' not found."); return None
            print(f"\nRetrieving file '{filename}'...")
            stored = self.files[filename]

            # Reconstruct data that was signed
            data_signed = number.long_to_bytes(stored['enc_content']) + filename.encode('utf-8')
            hash_obj_recomputed = SHA512.new(data_signed)

            # Verify signature first
            start=time.time()
            try: pkcs1_15.new(self.rsa_pub_key).verify(hash_obj_recomputed, stored['signature']); is_valid = True
            except (ValueError, TypeError): is_valid = False
            self.performance.record('RSA_Verify', time.time()-start)

            if not is_valid: print("  ❌ INVALID SIGNATURE! File may be tampered."); return None
            print("  ✓ Signature VALID.")

            # Decrypt content using Rabin
            start=time.time()
            decrypted_bytes = self._rabin_decrypt(stored['enc_content'], self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n'])
            dec_time = time.time()-start; self.performance.record('Rabin_Decrypt', dec_time)

            if decrypted_bytes is None: print("  ❌ Decryption FAILED."); return None

            print(f"  ✓ File decrypted successfully (Took: {dec_time:.6f}s)")
            return decrypted_bytes.decode('utf-8', errors='replace')


    def menu_secure_storage_rabin_rsa(): # System 26
        if not HAS_CRYPTO: print("\nSystem 26 requires 'pycryptodome'."); pause(); return

        system = SecureStorageRabinRSA()

        while True:
            print(f"\n{'='*70}\n  💾 SYSTEM 26: SECURE STORAGE (Rabin Enc + RSA Sign + SHA512)\n{'='*70}")
            print("  1. Store File (Encrypt & Sign)")
            print("  2. Retrieve File (Verify & Decrypt)")
            print("  3. List Stored Files")
            print("  4. Performance Analysis")
            print("  5. Compare Algs (Rabin vs RSA vs SHA512)")
            print("  6. Back to Main Menu"); print("-"*70)
            choice = input("Choice: ").strip()

            if choice == '1':
                filename = input("Filename to store: ")
                content = input("File content: ")
                system.store_file(filename, content)
            elif choice == '2':
                filename = input("Filename to retrieve: ")
                content = system.retrieve_file(filename)
                if content is not None: print(f"\n  Retrieved Content:\n{content}")
            elif choice == '3':
                print("\n--- Stored Files ---")
                if system.files: [print(f"  - {fname} (Enc: {str(finfo['enc_content'])[:20]}..., Sig: {finfo['signature'].hex()[:16]}...)") for fname, finfo in system.files.items()]
                else: print("  No files stored.")
            elif choice == '4': system.performance.print_graph("Secure Storage Performance")
            elif choice == '5': system.performance.compare_algorithms(['Rabin_Encrypt', 'RSA_Sign', 'SHA512_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_secure_storage_rabin_rsa(): print("\nSystem 26: Requires 'pycryptodome'."); pause()

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 4: BENCHMARK & MASTER MENU (Renumbered)
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 27: Universal Benchmark ---
# (Code for run_comprehensive_benchmark() from V4 goes here)
# Renumber System 23 -> 27
def run_comprehensive_benchmark(): # System 27
    # (Code from V4)
    print(f"\n{'='*70}\n  ⏱️ SYSTEM 27: COMPREHENSIVE ALGORITHM BENCHMARK\n{'='*70}")
    test_data_base = "The quick brown fox jumps over the lazy dog. " * 100; test_data_bytes = test_data_base.encode('utf-8'); data_size_kb = len(test_data_bytes) / 1024
    results = {}; tracker = PerformanceTracker()
    print(f"\nBenchmarking with data size: {data_size_kb:.2f} KB...")
    iter_fast, iter_med, iter_slow = 1000, 100, 10

    # Symmetric
    if HAS_CRYPTO:
        print("  Testing DES..."); des_key = get_random_bytes(8); cipher = DES.new(des_key, DES.MODE_ECB); start = time.time(); [cipher.encrypt(pad(test_data_bytes[:1024], DES.block_size)) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['DES Enc (1KB)'] = avg_time; tracker.record('Benchmark_DES_Enc', avg_time, 1024)
        print("  Testing AES-256..."); aes_key = get_random_bytes(32); cipher = AES.new(aes_key, AES.MODE_ECB); start = time.time(); [cipher.encrypt(pad(test_data_bytes[:1024], AES.block_size)) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['AES-256 Enc (1KB)'] = avg_time; tracker.record('Benchmark_AES256_Enc', avg_time, 1024)
        print("  Testing AES-GCM Encrypt..."); cipher_gcm = AES.new(aes_key, AES.MODE_GCM); start = time.time(); [cipher_gcm.encrypt_and_digest(test_data_bytes[:1024]) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['AES-GCM Enc (1KB)'] = avg_time; tracker.record('Benchmark_AESGCM_Enc', avg_time, 1024)

    # Asymmetric Enc/Dec
    if HAS_CRYPTO:
        print("  Testing RSA-2048 Encrypt..."); rsa_key = RSA.generate(2048); cipher_rsa = PKCS1_OAEP.new(rsa_key.publickey()); data_to_enc_rsa = test_data_bytes[:128]; start = time.time(); [cipher_rsa.encrypt(data_to_enc_rsa) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['RSA-2048 Enc (128B)'] = avg_time; tracker.record('Benchmark_RSA2048_Enc', avg_time, 128)
        print("  Testing RSA-2048 Decrypt..."); enc_rsa = cipher_rsa.encrypt(data_to_enc_rsa); cipher_rsa_priv = PKCS1_OAEP.new(rsa_key); start = time.time(); [cipher_rsa_priv.decrypt(enc_rsa) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['RSA-2048 Dec (128B)'] = avg_time; tracker.record('Benchmark_RSA2048_Dec', avg_time, 128)
        # Add Rabin benchmark if desired (using SecureStorageRabinRSA methods)
        # Add ElGamal Enc benchmark if desired (using HomomorphicProductElGamal methods)

    if HAS_PAILLIER:
        print("  Testing Paillier Encrypt..."); paillier_pub, paillier_priv = paillier.generate_paillier_keypair(n_length=1024); val_to_enc = 123456789; start = time.time(); [paillier_pub.encrypt(val_to_enc) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['Paillier Enc (1024b)'] = avg_time; tracker.record('Benchmark_Paillier_Enc', avg_time)
        print("  Testing Paillier Decrypt..."); enc_paillier = paillier_pub.encrypt(val_to_enc); start = time.time(); [paillier_priv.decrypt(enc_paillier) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['Paillier Dec (1024b)'] = avg_time; tracker.record('Benchmark_Paillier_Dec', avg_time)
        print("  Testing Paillier Add..."); enc1 = paillier_pub.encrypt(10); enc2 = paillier_pub.encrypt(20); start = time.time(); [enc1 + enc2 for _ in range(iter_fast)]; avg_time = (time.time() - start) / iter_fast; results['Paillier Add (1024b)'] = avg_time; tracker.record('Benchmark_Paillier_Add', avg_time)

    # Hashing
    if HAS_CRYPTO:
        print("  Testing MD5..."); start = time.time(); [MD5.new(test_data_bytes).hexdigest() for _ in range(iter_fast)]; avg_time = (time.time() - start) / iter_fast; results[f'MD5 Hash ({data_size_kb:.1f}KB)'] = avg_time; tracker.record('Benchmark_MD5', avg_time, len(test_data_bytes))
        print("  Testing SHA-256..."); start = time.time(); [SHA256.new(test_data_bytes).hexdigest() for _ in range(iter_fast)]; avg_time = (time.time() - start) / iter_fast; results[f'SHA-256 Hash ({data_size_kb:.1f}KB)'] = avg_time; tracker.record('Benchmark_SHA256', avg_time, len(test_data_bytes))
        print("  Testing SHA-512..."); start = time.time(); [SHA512.new(test_data_bytes).hexdigest() for _ in range(iter_fast)]; avg_time = (time.time() - start) / iter_fast; results[f'SHA-512 Hash ({data_size_kb:.1f}KB)'] = avg_time; tracker.record('Benchmark_SHA512', avg_time, len(test_data_bytes))

    # Signing / Verification
    if HAS_CRYPTO:
        # RSA
        hash_obj_sha256 = SHA256.new(test_data_bytes); signer_rsa = pkcs1_15.new(rsa_key); verifier_rsa = pkcs1_15.new(rsa_key.publickey())
        print("  Testing RSA-2048 Sign..."); start = time.time(); [signer_rsa.sign(hash_obj_sha256) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['RSA-2048 Sign (SHA256)'] = avg_time; tracker.record('Benchmark_RSA2048_Sign', avg_time)
        signature_rsa = signer_rsa.sign(hash_obj_sha256); print("  Testing RSA-2048 Verify..."); start = time.time(); [verifier_rsa.verify(hash_obj_sha256, signature_rsa) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['RSA-2048 Verify (SHA256)'] = avg_time; tracker.record('Benchmark_RSA2048_Verify', avg_time)
        # ElGamal
        print("  Testing ElGamal Sign..."); elg_sig_keys = generate_elgamal_sig_keys(bits=1024); hash_digest = SHA256.new(test_data_bytes).digest(); start = time.time(); [elgamal_sign(hash_digest, elg_sig_keys) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['ElGamal Sign (1024b)'] = avg_time; tracker.record('Benchmark_ElGamal_Sign', avg_time)
        elg_sig = elgamal_sign(hash_digest, elg_sig_keys); elg_pub = {k:v for k,v in elg_sig_keys.items() if k != 'x'}; print("  Testing ElGamal Verify..."); start = time.time(); [elgamal_verify(hash_digest, elg_sig, elg_pub) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['ElGamal Verify (1024b)'] = avg_time; tracker.record('Benchmark_ElGamal_Verify', avg_time)

    print("\n" + "="*70 + "\n  📊 BENCHMARK RESULTS\n" + "="*70)
    benchmark_tracker = PerformanceTracker(); benchmark_tracker.metrics = tracker.metrics
    benchmark_tracker.compare_algorithms(list(results.keys())) # Display using the comparison function
    print("\nBenchmark complete."); pause()


# --- Master Menu ---
def master_menu():
    """Master menu to select systems"""
    while True:
        clear_screen()
        print("\n" + "="*70)
        print("  💻 ICT3141 ULTIMATE EXAM SCRIPT V5 - MASTER MENU")
        print("="*70)

        # Dynamically generate menu based on available libraries
        print("\n--- PART 1: BASE 3-ALGORITHM SYSTEMS ---")
        print("  1-10. [Standard Combinations - See Previous Script]")
        print(f" 11. E-Voting (Paillier + ElGamal Sig){'' if HAS_PAILLIER else ' [DISABLED]'}")
        print(f" 12. Hybrid (Hill + RSA){'' if HAS_NUMPY else ' [DISABLED]'}")

        print("\n--- PART 2: INTERACTIVE CLASSICAL CIPHERS ---")
        print(" 13. Additive (Caesar)  | 14. Multiplicative")
        print(" 15. Affine             | 16. Vigenère")
        print(" 17. Autokey            | 18. Playfair")
        print(f" 19. Hill Cipher (2x2){'' if HAS_NUMPY else ' [DISABLED]'}")
        print(" 20. Transposition (Col)")

        print("\n--- PART 3: EXAM-SPECIFIC & NEW COMBINATIONS ---")
        print(f" 21. Payment Gateway (Paillier + RSA Sign + SHA256){'' if HAS_PAILLIER else ' [DISABLED]'}")
        print(f" 22. Secure Aggregation (Paillier + ElGamal Sign + SHA512){'' if HAS_PAILLIER else ' [DISABLED]'}")
        print(f" 23. Homomorphic Product (ElGamal Enc + RSA Sign + SHA256){'' if HAS_CRYPTO else ' [DISABLED]'}")
        print(f" 24. Secure Aggregation (Paillier + RSA Sign + SHA512){'' if HAS_PAILLIER else ' [DISABLED]'}")
        print(f" 25. Secure Transmission (AES-GCM + ElGamal Sign + SHA256){'' if HAS_CRYPTO else ' [DISABLED]'}")
        print(f" 26. Secure Storage (Rabin Enc + RSA Sign + SHA512){'' if HAS_CRYPTO else ' [DISABLED]'}")

        print("\n--- PART 4: TOOLS & EXIT ---")
        print(" 27. Universal Algorithm Benchmark")
        print(" 28. About This Script")
        print(" 29. Exit")
        print("-"*70)

        choice = input("\nSelect system (1-29): ").strip()

        # Map choices to functions (adjust numbers based on final count)
        menu_map = {
            '1': menu_email_system, '2': menu_banking_system, '3': menu_cloud_system,
            '4': menu_legacy_banking, '5': menu_healthcare, '6': menu_document_management,
            '7': menu_messaging, '8': menu_file_transfer, '9': menu_digital_library,
            '10': menu_secure_chat, '11': menu_voting_system, '12': menu_hill_hybrid,
            # Part 2
            '13': menu_additive_cipher, '14': menu_multiplicative_cipher, '15': menu_affine_cipher,
            '16': menu_vigenere_cipher, '17': menu_autokey_cipher, '18': menu_playfair_cipher,
            '19': menu_hill_cipher, '20': menu_transposition_cipher,
            # Part 3
            '21': menu_payment_gateway, '22': menu_secure_aggregation_paillier_elgamal,
            '23': menu_homomorphic_product_elgamal, '24': menu_secure_aggregation_paillier_rsa,
            '25': menu_secure_transmission_aes_elgamal, '26': menu_secure_storage_rabin_rsa,
            # Part 4
            '27': run_comprehensive_benchmark
        }

        if choice in menu_map:
             # Check for disabled options before calling
             func_to_call = menu_map[choice]
             # Crude check based on function names and required libs
             if ("hill" in func_to_call.__name__ or "affine_playfair" in func_to_call.__name__) and not HAS_NUMPY:
                 print(f"System {choice} requires NumPy (not found).")
                 pause()
             elif ("paillier" in func_to_call.__name__ or "voting" in func_to_call.__name__ or "gateway" in func_to_call.__name__) and not HAS_PAILLIER:
                 print(f"System {choice} requires 'phe' library (not found).")
                 pause()
             elif ("elgamal" in func_to_call.__name__ or "rsa" in func_to_call.__name__ or "aes" in func_to_call.__name__ or "des" in func_to_call.__name__ or "rabin" in func_to_call.__name__) and not HAS_CRYPTO:
                  # Check might be too broad, but covers most crypto-dependent ones
                  print(f"System {choice} requires 'pycryptodome' (not found).")
                  pause()
             else:
                  func_to_call()
        elif choice == '28': # About
             clear_screen()
             print("\n" + "="*70 + "\n  ABOUT THIS SCRIPT\n" + "="*70)
             print("\n  ICT3141 Ultimate Exam Script V5")
             print("  - Part 1: Base 3-Algorithm Systems (1-12)")
             print("  - Part 2: Interactive Classical Ciphers (13-20)")
             print("  - Part 3: Exam Scenarios & New Combinations (21-26)")
             print("  - Part 4: Benchmark & Utilities (27-29)")
             print("\n  Features: Menus, Performance Tracking, Graphs (ASCII/Matplotlib)")
             print("="*70); pause()
        elif choice == '29': # Exit
             print("\nGood luck with your exam! 🍀"); break
        else: print("Invalid choice."); pause()

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        master_menu()
    except KeyboardInterrupt: print("\n\nExiting."); sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 An unexpected error occurred: {e}\n")
        import traceback
        traceback.print_exc()
        print("\nEnsure all libraries (pycryptodome, numpy, phe, matplotlib) are installed.")
        sys.exit(1)
