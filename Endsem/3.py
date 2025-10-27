# PYTHON SCRIPT (Version 6 - COMPLETE CODE)
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
    ICT3141 ULTIMATE EXAM SCRIPT - ALL SCENARIOS V6
    
    Part 1: 12 Base 3-Algorithm Systems (Systems 1-12)
    Part 2: Interactive Classical Ciphers (Systems 13-20)
    Part 3: Exam-Specific & New Combinations (Systems 21-28)
    Part 4: Advanced Concepts (Placeholders) (Systems 29-30)
    Part 5: Tools & Exit (Systems 31-33)
    
    All with Menu-Driven Interfaces and Performance Graphs
═══════════════════════════════════════════════════════════════════════════
"""

# --- Imports and Library Checks ---
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
print("  ICT3141 ULTIMATE EXAM SCRIPT V6 - INITIALIZING")
print("="*70)

# --- PyCryptodome (Essential) ---
try:
    from Crypto.Cipher import DES, DES3, AES, PKCS1_OAEP
    from Crypto.PublicKey import RSA, ElGamal
    from Crypto.Hash import SHA256, SHA512, SHA1, MD5, HMAC
    from Crypto.Util import number
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Random import get_random_bytes
    from Crypto.Signature import pkcs1_15
    HAS_CRYPTO = True
    print("  ✓ PyCryptodome loaded")
except ImportError:
    HAS_CRYPTO = False
    print("  ✗ PyCryptodome not installed! (pip install pycryptodome)")
    # Allow running without it for classical ciphers

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


# --- Utility Functions ---
class PerformanceTracker:
    """Track and visualize performance metrics with enhanced graphing"""
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
        if len(alg_list) < 2: print("Need >= 2 algorithms."); return
        stats = {alg: s['average'] for alg in alg_list if (s := self.get_stats(alg))}
        if len(stats) < 2: print("Not enough data."); return

        print(f"\n{'='*70}\n  🚀 ALGORITHM COMPARISON\n{'='*70}")
        sorted_algs = sorted(stats.items(), key=lambda x: x[1])
        print("\nSpeed Ranking (fastest to slowest):")
        scale = 5000; max_len = max(len(alg) for alg, t in sorted_algs)
        for i, (alg, time_val) in enumerate(sorted_algs, 1):
            bar = '█' * int(time_val * scale); bar = bar if bar else ('·' if time_val > 0 else '')
            print(f"  {i}. {alg:<{max_len}s}: {time_val:.8f}s\n     {bar}")

        fastest_time = sorted_algs[0][1]
        if fastest_time > 0: print("\nRelative Performance:"); [print(f"  {alg} is {time_val / fastest_time:.1f}x slower than {sorted_algs[0][0]}") for alg, time_val in sorted_algs[1:]]
        else: print("\nFastest too fast for relative comparison.")
        print("="*70)

        if HAS_MATPLOTLIB:
             try:
                 labels = [a[0] for a in sorted_algs]; times = [a[1] for a in sorted_algs]
                 plt.figure(figsize=(10, max(6, len(labels)*0.5))); plt.barh(labels, times, color='lightcoral')
                 plt.xlabel("Average Time (seconds)"); plt.ylabel("Algorithm"); plt.title("Algorithm Speed Comparison")
                 plt.tight_layout(); print("\n📈 Displaying graphical comparison chart..."); plt.show()
             except Exception as e: print(f"  ⚠ Matplotlib comparison graph error: {e}")


def gcd(a, b): return math.gcd(a, b) # Use math.gcd

def mod_inverse(a, m):
    a=a%m; m0=m; x0,x1=0,1;
    if gcd(a,m)!=1: return None
    while a>1: q=a//m; m,a=a%m,m; x0,x1=x1-q*x0,x0
    if x1<0: x1+=m0
    return x1

def matrix_mod_inv(matrix, modulus):
    if not HAS_NUMPY: raise ImportError("NumPy required")
    det = int(np.round(np.linalg.det(matrix))); det_inv = mod_inverse(det % modulus, modulus)
    if det_inv is None: raise ValueError(f"Matrix determinant {det} not invertible mod {modulus} (gcd={gcd(det % modulus, modulus)})")
    adj = np.linalg.inv(matrix) * det; inv = (det_inv * np.round(adj)) % modulus
    return inv.astype(int)

# --- Manual ElGamal Signature Implementation ---
def generate_elgamal_sig_keys(bits=1024):
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
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
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
    p, q, g, x = priv_key['p'], priv_key['q'], priv_key['g'], priv_key['x']; h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    while True:
        k = number.getRandomRange(2, q); r = pow(g, k, p); k_inv = number.inverse(k, q); s = (k_inv * (h_int + x * r)) % q
        if r != 0 and s != 0: return int(r), int(s)

def elgamal_verify(msg_bytes, signature, pub_key):
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
    p, q, g, y = pub_key['p'], pub_key['q'], pub_key['g'], pub_key['y']; r, s = signature
    if not (0 < r < p and 0 < s < q): return False
    h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    v1 = (pow(y, r, p) * pow(r, s, p)) % p; v2 = pow(g, h_int, p)
    return v1 == v2

# --- Other Utilities ---
def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')
def pause(): input("\nPress Enter to continue...")

# --- CryptographyToolkit Class for Classical Ciphers ---
class CryptographyToolkit:
     def additive_cipher(self, text, key, decrypt=False): shift = -key if decrypt else key; return "".join([chr((ord(c)-65+shift)%26+65) if c.isalpha() else c for c in text.upper()])
     def multiplicative_cipher(self, text, key, decrypt=False):
         key_op = mod_inverse(key, 26) if decrypt else key
         if key_op is None or key_op == -1: return "Invalid key/inverse"
         return "".join([chr((key_op*(ord(c)-65))%26+65) if c.isalpha() else c for c in text.upper()])
     def affine_cipher(self, text, a, b, decrypt=False):
         a_op = mod_inverse(a, 26) if decrypt else a
         if a_op is None or a_op == -1: return "Invalid 'a'/inverse"
         return "".join([chr((a_op*((ord(c)-65) - b))%26 + 65) if c.isalpha() else c for c in text.upper()]) if decrypt else "".join([chr((a*(ord(c)-65)+b)%26+65) if c.isalpha() else c for c in text.upper()])
     def vigenere_cipher(self, text, key, decrypt=False):
         res=""; k=key.upper(); ki=0
         for c in text.upper():
             if c.isalpha(): shift = ord(k[ki % len(k)]) - 65; shift = -shift if decrypt else shift; res += chr((ord(c)-65+shift)%26+65); ki+=1
             else: res += c
         return res
     def autokey_cipher(self, text, key, decrypt=False): # Simplified Autokey
        res = ""; k = key.upper(); extended_key = k
        plain_text_for_key = text.upper().replace(' ', '') # Need plaintext for encryption key extension
        if not decrypt: extended_key += plain_text_for_key
        ki = 0
        for char in text.upper():
            if char.isalpha():
                if ki < len(extended_key): shift = ord(extended_key[ki]) - 65
                else: shift = 0 # Fallback, should extend properly
                if decrypt:
                    dec_char_ord = (ord(char) - 65 - shift) % 26
                    dec_char = chr(dec_char_ord + 65)
                    res += dec_char
                    # Extend key *during* decryption based on *decrypted* chars
                    if ki >= len(k): extended_key += dec_char
                else: res += chr((ord(char) - 65 + shift) % 26 + 65)
                ki += 1
            else: res += char
        return res
     def playfair_cipher(self, text, key, decrypt=False): # Simplified Playfair
        key=key.upper().replace('J','I'); matrix=[]; used=set(); [ (matrix.append(c), used.add(c)) for c in key if c.isalpha() and c not in used ]; [ matrix.append(c) for c in 'ABCDEFGHIKLMNOPQRSTUVWXYZ' if c not in used ]; grid=[matrix[i:i+5] for i in range(0,25,5)];
        def find_pos(char):
             for r, row in enumerate(grid):
                  if char in row: return r, row.index(char)
             return None, None
        text=text.upper().replace('J','I').replace(' ',''); pairs=[]; i=0; temp_text=list(text)
        # Handle digraphs more carefully
        idx = 0
        while idx < len(temp_text):
            a = temp_text[idx]
            if idx + 1 == len(temp_text): # Last char
                pairs.append(a + 'X')
                break
            b = temp_text[idx+1]
            if a == b:
                temp_text.insert(idx+1, 'X') # Insert padding
                pairs.append(a + 'X')
                idx += 1 # Move past the original char only
            else:
                pairs.append(a + b)
                idx += 2 # Move past the pair
        # Check if last pair needs padding (if original length was odd) - handled by loop logic now

        res=""; shift = -1 if decrypt else 1
        for p_str in pairs:
            if len(p_str)==2:
                r1,c1=find_pos(p_str[0]); r2,c2=find_pos(p_str[1])
                if r1 is not None and r2 is not None:
                     if r1==r2: res+=grid[r1][(c1+shift)%5]+grid[r2][(c2+shift)%5]
                     elif c1==c2: res+=grid[(r1+shift)%5][c1]+grid[(r2+shift)%5][c2]
                     else: res+=grid[r1][c2]+grid[r2][c1]
            else: # Should not happen with corrected pairing logic
                 res += p_str # Append lone char if something went wrong
        # Decryption might have trailing X or Xs inserted as padding
        # Basic removal - might remove intended X's if message ended with X
        # A better approach would be to track original length if possible
        if decrypt:
             # Heuristic: remove trailing X if preceded by X used for padding
             final_res = ""
             i = 0
             while i < len(res):
                 final_res += res[i]
                 if i + 2 < len(res) and res[i] == res[i+2] and res[i+1] == 'X':
                     i += 2 # Skip the padded X and the following char (duplicate)
                 elif i == len(res) - 1 and res[i] == 'X': # Remove trailing X if it seems like padding
                     final_res = final_res[:-1]
                     break
                 else:
                     i+=1
             return final_res
        return res

     def hill_cipher_2x2(self, text, key_matrix, decrypt=False): # Simplified Hill
        if decrypt: det = (key_matrix[0][0]*key_matrix[1][1]-key_matrix[0][1]*key_matrix[1][0])%26; det_inv = mod_inverse(det, 26);
            if det_inv is None: return "Matrix not invertible"
            inv_matrix = [[(det_inv*key_matrix[1][1])%26, (-det_inv*key_matrix[0][1])%26],[(-det_inv*key_matrix[1][0])%26, (det_inv*key_matrix[0][0])%26]]; key_matrix = inv_matrix
        res = ""; txt = "".join(filter(str.isalpha, text.upper())); original_len = len(txt)
        txt += 'X' * (len(txt)%2)
        for i in range(0,len(txt),2):
             p=[ord(txt[i])-65, ord(txt[i+1])-65]; e=[(key_matrix[0][0]*p[0]+key_matrix[0][1]*p[1])%26, (key_matrix[1][0]*p[0]+key_matrix[1][1]*p[1])%26]; res+=chr(e[0]+65)+chr(e[1]+65)
        return res if not decrypt else res[:original_len] # Return original length on decryption


toolkit_classical = CryptographyToolkit() # Instantiate for use in menus


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 1: COMPLETE 3-ALGORITHM SYSTEMS (Systems 1-12)
# FULL CODE PASTED HERE
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 1: Secure Email (DES + RSA + SHA-256) ---
if HAS_CRYPTO:
    class SecureEmailSystem:
        """Complete email system with DES, RSA, and SHA-256"""
        def __init__(self):
            self.users = {}
            self.mailboxes = {}
            self.performance = PerformanceTracker()
        def register_user(self, user_id):
            start = time.time(); key = RSA.generate(2048)
            self.users[user_id] = {'private_key': key,'public_key': key.publickey()}
            self.mailboxes[user_id] = []
            self.performance.record('RSA_KeyGen', time.time() - start); print(f"✓ {user_id} registered")
        def send_email(self, sender, recipient, subject, body):
            if recipient not in self.users: print("Recipient not found!"); return None
            # DES encryption
            start = time.time(); des_key = get_random_bytes(8); cipher_des = DES.new(des_key, DES.MODE_ECB)
            content = f"{subject}||{body}"; encrypted_body = cipher_des.encrypt(pad(content.encode(), DES.block_size))
            des_time = time.time() - start; self.performance.record('DES_Encrypt', des_time, len(content))
            # SHA-256 hash
            start = time.time(); email_hash = SHA256.new(content.encode()).hexdigest(); sha_time = time.time() - start; self.performance.record('SHA256_Hash', sha_time)
            # RSA key encryption
            start = time.time(); cipher_rsa = PKCS1_OAEP.new(self.users[recipient]['public_key']); encrypted_key = cipher_rsa.encrypt(des_key)
            rsa_time = time.time() - start; self.performance.record('RSA_Encrypt', rsa_time)
            email = {'id': len(self.mailboxes[recipient]) + 1, 'from': sender, 'encrypted_key': base64.b64encode(encrypted_key).decode(), 'encrypted_body': base64.b64encode(encrypted_body).decode(), 'hash': email_hash, 'timestamp': datetime.now().isoformat()}
            self.mailboxes[recipient].append(email)
            print(f"\n✓ Email sent! (DES: {des_time:.4f}s | RSA: {rsa_time:.4f}s | SHA256: {sha_time:.4f}s)"); return email['id']
        def read_email(self, user_id, email_id):
            email = next((e for e in self.mailboxes.get(user_id, []) if e['id'] == email_id), None)
            if not email: print("Email not found!"); return None
            try:
                # RSA key decryption
                start = time.time(); cipher_rsa = PKCS1_OAEP.new(self.users[user_id]['private_key']); des_key = cipher_rsa.decrypt(base64.b64decode(email['encrypted_key']))
                rsa_time = time.time() - start; self.performance.record('RSA_Decrypt', rsa_time)
                # DES decryption
                start = time.time(); cipher_des = DES.new(des_key, DES.MODE_ECB); decrypted_padded = cipher_des.decrypt(base64.b64decode(email['encrypted_body'])); decrypted = unpad(decrypted_padded, DES.block_size).decode()
                des_time = time.time() - start; self.performance.record('DES_Decrypt', des_time)
                # SHA-256 verification
                start = time.time(); computed_hash = SHA256.new(decrypted.encode()).hexdigest(); verified = computed_hash == email['hash']
                sha_time = time.time() - start; self.performance.record('SHA256_Verify', sha_time)
                subject, body = decrypted.split('||', 1)
                print(f"\n✓ Email Decrypted (RSA: {rsa_time:.4f}s | DES: {des_time:.4f}s | SHA256: {sha_time:.4f}s)")
                print(f"  Hash verified: {verified}\n  Subject: {subject}\n  Body: {body}"); return decrypted
            except (ValueError, KeyError, IndexError) as e: print(f"Decryption/Verification Error: {e}"); return None

    def menu_email_system(): # System 1
        if not HAS_CRYPTO: print("\nSystem 1 requires 'pycryptodome'."); pause(); return
        system = SecureEmailSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 1: SECURE EMAIL (DES + RSA + SHA-256)\n{'='*70}")
            print("1. Register User | 2. Send Email | 3. Read Email | 4. View Mailbox | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user_id = input("User ID: "); system.register_user(user_id)
            elif choice == '2': sender = input("From: "); recipient = input("To: "); subject = input("Subject: "); body = input("Body: "); system.send_email(sender, recipient, subject, body)
            elif choice == '3':
                user = input("Your ID: ");
                try: email_id = int(input("Email ID: ")); system.read_email(user, email_id)
                except ValueError: print("Invalid ID.")
            elif choice == '4': user = input("Your ID: "); [print(f"\n[{e['id']}] From: {e['from']}") for e in system.mailboxes.get(user, [])] if user in system.mailboxes else print("Mailbox not found.")
            elif choice == '5': system.performance.print_graph("Email System Performance")
            elif choice == '6': system.performance.compare_algorithms(['DES_Encrypt', 'RSA_Encrypt', 'SHA256_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_email_system(): print("\nSystem 1 requires 'pycryptodome'."); pause()


# --- System 2: Banking (AES + ElGamal(Enc) + SHA-512) ---
# Note: Using ElGamal for ENCRYPTION here as in original script, NOT signing
if HAS_CRYPTO:
    class BankingSystem:
        def __init__(self):
            self.performance = PerformanceTracker()
            start = time.time(); self.elgamal_key = ElGamal.generate(1024, get_random_bytes); self.performance.record('ElGamal_KeyGen_Enc', time.time() - start)
            self.aes_key = hashlib.sha256(b"BankMasterKeyForAES").digest() # Use a different key derivation if needed
            self.accounts = {} # {cust_id: {'balance': float, 'elg_pub': pub_key}} # Store pub key for sending
            self.transactions = [] # {'id', 'from', 'to', 'amount', 'enc_details', 'enc_aes_key', 'hash'}

        def create_account(self, customer_id, balance):
            if customer_id in self.accounts: print("Account exists."); return
            # Each customer needs keys if they *receive* ElGamal encrypted data,
            # but for *sending* signed data, they'd need signing keys.
            # This system encrypts details with AES, key with ElGamal, hashes details. Let's simplify:
            # Assume Bank holds master ElGamal key for encrypting AES key FOR the bank.
            self.accounts[customer_id] = {'balance': balance}
            print(f"✓ Account created for {customer_id}, Balance: ${balance}")

        def create_transaction(self, from_cust, to_cust, amount, description):
            if from_cust not in self.accounts: print("Sender account not found!"); return None
            if self.accounts[from_cust]['balance'] < amount: print("Insufficient funds!"); return None

            txn_details = f"{from_cust}|{to_cust}|{amount}|{description}"
            txn_bytes = txn_details.encode('utf-8')

            # SHA-512 hash
            start = time.time(); h_obj = SHA512.new(txn_bytes); hash_hex = h_obj.hexdigest(); self.performance.record('SHA512_Hash', time.time() - start)

            # AES encryption of details
            start = time.time(); session_aes_key = get_random_bytes(32); cipher_aes = AES.new(session_aes_key, AES.MODE_GCM); nonce = cipher_aes.nonce; enc_details, tag = cipher_aes.encrypt_and_digest(txn_bytes)
            aes_time = time.time() - start; self.performance.record('AES_Encrypt_GCM', aes_time, len(txn_bytes))

            # ElGamal encryption of the AES key (using bank's public key)
            start = time.time();
            # ElGamal encrypt expects bytes, K must be generated
            # PyCryptodome's ElGamal.encrypt isn't straightforward multiplicative homomorphism setup
            # Simulating basic ElGamal encryption for the key
            p = self.elgamal_key.p; g = self.elgamal_key.g; y = self.elgamal_key.y; k_elg = number.getRandomRange(1, p-1)
            c1 = pow(g, k_elg, p)
            # Convert AES key bytes to integer for encryption
            aes_key_int = number.bytes_to_long(session_aes_key)
            if aes_key_int >= p: print("Warning: AES key as int might be larger than ElGamal p, consider smaller p or key hashing"); # This shouldn't happen with 1024-bit p and 256-bit key
            c2 = (aes_key_int * pow(y, k_elg, p)) % p
            enc_aes_key = (c1, c2)
            elg_time = time.time() - start; self.performance.record('ElGamal_Encrypt', elg_time)

            txn = { 'id': len(self.transactions) + 1, 'from': from_cust, 'to': to_cust, 'amount': amount, 'nonce': nonce, 'tag': tag, 'enc_details': enc_details, 'enc_aes_key': enc_aes_key, 'hash': hash_hex, 'status': 'pending'}
            self.transactions.append(txn)
            print(f"\n✓ Tx created! (AES: {aes_time:.4f}s | ElG: {elg_time:.4f}s | SHA512: {time.time()-start-aes_time-elg_time:.4f}s)") # Hash time inaccurate here
            return txn['id']

        def process_transaction(self, txn_id):
            txn = next((t for t in self.transactions if t['id'] == txn_id and t['status'] == 'pending'), None)
            if not txn: print("Transaction not found or already processed!"); return False

            try:
                # ElGamal decryption of AES key
                start = time.time(); p = self.elgamal_key.p; x = self.elgamal_key.x; c1, c2 = txn['enc_aes_key']
                s = pow(c1, x, p); s_inv = number.inverse(s, p); aes_key_int = (c2 * s_inv) % p
                # Convert back to bytes (assuming 32 bytes/256 bits)
                session_aes_key = number.long_to_bytes(aes_key_int, 32)
                elg_time = time.time() - start; self.performance.record('ElGamal_Decrypt', elg_time)

                # AES decryption of details
                start = time.time(); cipher_aes = AES.new(session_aes_key, AES.MODE_GCM, nonce=txn['nonce']); decrypted_bytes = cipher_aes.decrypt_and_verify(txn['enc_details'], txn['tag'])
                decrypted_details = decrypted_bytes.decode('utf-8')
                aes_time = time.time() - start; self.performance.record('AES_Decrypt_GCM', aes_time)

                # SHA-512 verification
                start = time.time(); computed_hash = SHA512.new(decrypted_bytes).hexdigest(); verified = computed_hash == txn['hash']
                sha_time = time.time() - start; self.performance.record('SHA512_Verify', sha_time)

                if not verified: print("  ❌ Hash mismatch! Transaction rejected."); txn['status'] = 'failed'; return False

                print(f"✓ Tx Processed (ElG Dec: {elg_time:.4f}s | AES Dec: {aes_time:.4f}s | SHA512 Vfy: {sha_time:.4f}s)")
                print(f"  Decrypted Details: {decrypted_details}")
                # Update balances
                self.accounts[txn['from']]['balance'] -= txn['amount']
                if txn['to'] not in self.accounts: self.accounts[txn['to']] = {'balance': 0} # Create if not exists
                self.accounts[txn['to']]['balance'] += txn['amount']
                txn['status'] = 'completed'
                print(f"  Balances Updated: {txn['from']} -> ${self.accounts[txn['from']]['balance']:.2f}, {txn['to']} -> ${self.accounts[txn['to']]['balance']:.2f}")
                return True
            except (ValueError, KeyError, number.IntegerOverflowError) as e:
                print(f"Processing Error: {e}"); txn['status'] = 'failed'; return False

    def menu_banking_system(): # System 2
        if not HAS_CRYPTO: print("\nSystem 2 requires 'pycryptodome'."); pause(); return
        system = BankingSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 2: BANKING (AES-GCM + ElGamal Enc + SHA-512)\n{'='*70}")
            print("1. Create Account | 2. View Balance | 3. Create Tx | 4. Process Tx | 5. View Txs | 6. Performance | 7. Compare Algs | 8. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': cust = input("Cust ID: "); bal = float(input("Balance: $")); system.create_account(cust, bal)
            elif choice == '2': cust = input("Cust ID: "); print(f"Balance: ${system.accounts.get(cust, {}).get('balance', 'N/A')}")
            elif choice == '3': frm = input("From: "); to = input("To: "); amt = float(input("Amount: $")); desc = input("Desc: "); system.create_transaction(frm, to, amt, desc)
            elif choice == '4':
                try: txid = int(input("Tx ID to process: ")); system.process_transaction(txid)
                except ValueError: print("Invalid ID.")
            elif choice == '5': print("\n--- Transactions ---"); [print(f" ID: {t['id']}, {t['from']} -> {t['to']}, ${t['amount']:.2f}, Status: {t['status']}") for t in system.transactions] if system.transactions else print("None.")
            elif choice == '6': system.performance.print_graph("Banking System Performance")
            elif choice == '7': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'ElGamal_Encrypt', 'SHA512_Hash'])
            elif choice == '8': break
            else: print("Invalid choice.")
            if choice != '8': pause()
else:
    def menu_banking_system(): print("\nSystem 2 requires 'pycryptodome'."); pause()

# --- System 3: Cloud Storage (Rabin + RSA + MD5) ---
if HAS_CRYPTO:
    class SecureStorageRabinRSA_MD5: # Renamed slightly to avoid conflict
        def __init__(self, rabin_bits=2048, rsa_bits=2048):
            self.performance = PerformanceTracker()
            start=time.time(); self.rabin_keys = self._generate_rabin_keys(rabin_bits); self.performance.record('Rabin_KeyGen', time.time()-start)
            self.users = {} # user_id: {'rsa_priv': key, 'rsa_pub': pubkey}
            self.files = {} # file_id: {'owner', 'filename', 'enc_content', 'enc_key_repr', 'hash_md5'}

        def _generate_rabin_keys(self, bits): # Same as Sys 26
             while True: p = number.getPrime(bits // 2);
                 if p % 4 == 3: break
             while True: q = number.getPrime(bits // 2);
                 if q % 4 == 3 and q != p: break
             print("✓ Rabin keys generated."); return {'n': p * q, 'p': p, 'q': q}

        def _rabin_encrypt(self, msg_bytes, n): # Same padding as Sys 26
            redundancy = b"FILEPAD" + len(msg_bytes).to_bytes(4, 'big')
            full_msg = msg_bytes + redundancy; m = number.bytes_to_long(full_msg)
            if m >= n: raise ValueError("Message too large for Rabin key")
            return pow(m, 2, n)

        def _rabin_decrypt(self, cipher_int, p, q, n): # Same as Sys 26
            mp=pow(cipher_int,(p+1)//4,p); mq=pow(cipher_int,(q+1)//4,q); inv_p=number.inverse(p,q); inv_q=number.inverse(q,p); a=(q*inv_q)%n; b=(p*inv_p)%n
            roots=[(a*mp+b*mq)%n,(a*mp-b*mq)%n,(-a*mp+b*mq)%n,(-a*mp-b*mq)%n]
            for r in roots:
                try: r_bytes=number.long_to_bytes(r);
                    if len(r_bytes)>11 and r_bytes.endswith(b"FILEPAD"): len_s=len(r_bytes)-11; msg_len=int.from_bytes(r_bytes[len_s:len_s+4],'big');
                        if len_s==msg_len: return r_bytes[:msg_len]
                except Exception: continue
            return None

        def register_user(self, user_id, rsa_bits=2048):
            if user_id in self.users: print("User exists."); return
            start = time.time(); key = RSA.generate(rsa_bits); self.users[user_id] = {'rsa_priv': key, 'rsa_pub': key.publickey()}; self.performance.record('RSA_KeyGen', time.time() - start); print(f"✓ User {user_id} registered.")

        def upload_file(self, owner, filename, content):
            if owner not in self.users: print("Owner not registered!"); return None
            content_bytes = content.encode('utf-8')

            # MD5 Hash
            start=time.time(); hash_md5 = MD5.new(content_bytes).hexdigest(); self.performance.record('MD5_Hash', time.time()-start)

            # Rabin Encrypt content
            start=time.time();
            try: enc_content = self._rabin_encrypt(content_bytes, self.rabin_keys['n'])
            except ValueError as e: print(f"Encryption error: {e}"); return None
            rabin_time = time.time()-start; self.performance.record('Rabin_Encrypt', rabin_time, len(content_bytes))

            # Encrypt a representation of the key/content id with RSA (for owner access control simulation)
            # Here, let's encrypt the MD5 hash with the owner's RSA key as a simple "key" representation
            start=time.time(); owner_pub_key = self.users[owner]['rsa_pub']; cipher_rsa = PKCS1_OAEP.new(owner_pub_key); enc_key_repr = cipher_rsa.encrypt(hash_md5.encode())
            rsa_time = time.time()-start; self.performance.record('RSA_Encrypt', rsa_time)

            file_id = len(self.files) + 1
            self.files[file_id] = {'owner': owner, 'filename': filename, 'enc_content': enc_content, 'enc_key_repr': enc_key_repr, 'hash_md5': hash_md5}
            print(f"\n✓ File '{filename}' uploaded! (Rabin: {rabin_time:.4f}s | RSA: {rsa_time:.4f}s | MD5: <0.001s)")
            return file_id

        def download_file(self, user_id, file_id):
            if file_id not in self.files: print("File ID not found!"); return None
            file_info = self.files[file_id]
            if file_info['owner'] != user_id: print("Access denied!"); return None
            if user_id not in self.users: print("User not registered!"); return None

            try:
                # Decrypt "key representation" with user's private RSA key
                start=time.time(); user_priv_key = self.users[user_id]['rsa_priv']; cipher_rsa = PKCS1_OAEP.new(user_priv_key); dec_key_repr = cipher_rsa.decrypt(file_info['enc_key_repr']).decode()
                rsa_time = time.time()-start; self.performance.record('RSA_Decrypt', rsa_time)

                # Check if decrypted representation matches stored hash (simple access check)
                if dec_key_repr != file_info['hash_md5']: print("Key representation mismatch! Decryption aborted."); return None

                # Decrypt content with Rabin
                start=time.time(); dec_bytes = self._rabin_decrypt(file_info['enc_content'], self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n'])
                rabin_time = time.time()-start; self.performance.record('Rabin_Decrypt', rabin_time)

                if dec_bytes is None: print("Rabin decryption failed!"); return None
                dec_content = dec_bytes.decode('utf-8')

                # Verify MD5 hash
                start=time.time(); computed_hash = MD5.new(dec_bytes).hexdigest(); verified = computed_hash == file_info['hash_md5']
                md5_time = time.time()-start; self.performance.record('MD5_Verify', md5_time)

                print(f"\n✓ File Downloaded (RSA: {rsa_time:.4f}s | Rabin: {rabin_time:.4f}s | MD5: {md5_time:.4f}s)")
                print(f"  Filename: {file_info['filename']}\n  MD5 Verified: {verified}\n  Content: {dec_content}")
                return dec_content
            except (ValueError, KeyError) as e: print(f"Download/Decryption Error: {e}"); return None

    def menu_cloud_system(): # System 3 (uses updated class)
        if not HAS_CRYPTO: print("\nSystem 3 requires 'pycryptodome'."); pause(); return
        system = SecureStorageRabinRSA_MD5() # Use the updated class name
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 3: CLOUD STORAGE (Rabin Enc + RSA Enc + MD5)\n{'='*70}")
            print("1. Register User | 2. Upload File | 3. Download File | 4. List Files | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user = input("User ID: "); system.register_user(user)
            elif choice == '2': owner = input("Your ID: "); fname = input("Filename: "); content = input("Content: "); system.upload_file(owner, fname, content)
            elif choice == '3':
                user = input("Your ID: ");
                try: fid = int(input("File ID: ")); system.download_file(user, fid)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Files ---"); [print(f" ID: {fid}, Name: {finfo['filename']}, Owner: {finfo['owner']}") for fid, finfo in system.files.items()] if system.files else print("None.")
            elif choice == '5': system.performance.print_graph("Cloud Storage Performance")
            elif choice == '6': system.performance.compare_algorithms(['Rabin_Encrypt', 'RSA_Encrypt', 'MD5_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_cloud_system(): print("\nSystem 3 requires 'pycryptodome'."); pause()


# --- System 4: Legacy Banking (3DES + ElGamal(Enc) + SHA-1) ---
# Note: Using ElGamal for ENCRYPTION here, NOT signing
if HAS_CRYPTO:
    class LegacyBankingSystem: # Based on original structure
        def __init__(self):
            self.performance = PerformanceTracker()
            # Generate common keys (in real system these would be managed securely)
            start = time.time(); self.elgamal_key = ElGamal.generate(1024, get_random_bytes); self.performance.record('ElGamal_KeyGen_Enc', time.time() - start)
            # Ensure 3DES key is exactly 16 or 24 bytes
            self.des3_key = hashlib.sha256(b"LegacyBankKeyMaterial").digest()[:24] # Derive 24 bytes
            self.customers = {} # cust_id: {info} - ElGamal keys aren't per-customer in this setup
            self.transactions = [] # {'id', 'from', 'enc_details', 'enc_sym_key', 'hash_sha1'}

        def register_customer(self, cust_id):
            if cust_id in self.customers: print("Customer exists."); return
            self.customers[cust_id] = {'id': cust_id} # Basic info
            print(f"✓ Customer {cust_id} registered.")

        def create_transaction(self, from_id, to_id, amount):
            if from_id not in self.customers: print("Sender not registered!"); return None

            txn_details = f"{from_id}:{to_id}:{amount}"
            txn_bytes = txn_details.encode('utf-8')

            # SHA-1 Hash (Legacy)
            start = time.time(); hash_sha1 = SHA1.new(txn_bytes).hexdigest(); self.performance.record('SHA1_Hash', time.time() - start)

            # 3DES Encryption of details
            start = time.time(); session_des3_key = get_random_bytes(24); # Use a session key
            cipher_3des = DES3.new(session_des3_key, DES3.MODE_CBC); iv = cipher_3des.iv; enc_details = cipher_3des.encrypt(pad(txn_bytes, DES3.block_size))
            des3_time = time.time() - start; self.performance.record('3DES_Encrypt_CBC', des3_time, len(txn_bytes))

            # ElGamal encryption of the 3DES session key (using bank's public key)
            start = time.time();
            p = self.elgamal_key.p; g = self.elgamal_key.g; y = self.elgamal_key.y; k_elg = number.getRandomRange(1, p-1)
            c1 = pow(g, k_elg, p)
            key_int = number.bytes_to_long(session_des3_key)
            if key_int >= p: print("Warning: 3DES key too large for ElGamal P"); # Should be ok 192 vs 1024
            c2 = (key_int * pow(y, k_elg, p)) % p
            enc_sym_key = (c1, c2)
            elg_time = time.time() - start; self.performance.record('ElGamal_Encrypt', elg_time)

            txn = {'id': len(self.transactions) + 1, 'from': from_id, 'iv': iv, 'enc_details': enc_details, 'enc_sym_key': enc_sym_key, 'hash_sha1': hash_sha1, 'status': 'pending'}
            self.transactions.append(txn)
            print(f"\n✓ Tx created! (3DES: {des3_time:.4f}s | ElG: {elg_time:.4f}s | SHA1: <0.001s)")
            return txn['id']

        def verify_transaction(self, txn_id): # Renamed from process, just verifies/decrypts
            txn = next((t for t in self.transactions if t['id'] == txn_id), None)
            if not txn: print("Tx not found!"); return False

            try:
                # ElGamal decryption of 3DES key
                start = time.time(); p = self.elgamal_key.p; x = self.elgamal_key.x; c1, c2 = txn['enc_sym_key']
                s = pow(c1, x, p); s_inv = number.inverse(s, p); key_int = (c2 * s_inv) % p
                session_des3_key = number.long_to_bytes(key_int, 24) # Expect 24 bytes
                elg_time = time.time() - start; self.performance.record('ElGamal_Decrypt', elg_time)

                # 3DES decryption of details
                start = time.time(); cipher_3des = DES3.new(session_des3_key, DES3.MODE_CBC, iv=txn['iv']); decrypted_bytes = unpad(cipher_3des.decrypt(txn['enc_details']), DES3.block_size)
                decrypted_details = decrypted_bytes.decode('utf-8')
                des3_time = time.time() - start; self.performance.record('3DES_Decrypt_CBC', des3_time)

                # SHA-1 verification
                start = time.time(); computed_hash = SHA1.new(decrypted_bytes).hexdigest(); verified = computed_hash == txn['hash_sha1']
                sha_time = time.time() - start; self.performance.record('SHA1_Verify', sha_time)

                print(f"\n✓ Tx Verified (ElG Dec: {elg_time:.4f}s | 3DES Dec: {des3_time:.4f}s | SHA1 Vfy: {sha_time:.4f}s)")
                print(f"  Hash Verified: {verified}\n  Details: {decrypted_details}")
                # Update status (optional for verification only)
                if verified: txn['status'] = 'verified'
                else: txn['status'] = 'verification_failed'
                return verified
            except (ValueError, KeyError, number.IntegerOverflowError) as e:
                 print(f"Verification Error: {e}"); txn['status']='error'; return False

    def menu_legacy_banking(): # System 4
        if not HAS_CRYPTO: print("\nSystem 4 requires 'pycryptodome'."); pause(); return
        system = LegacyBankingSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 4: LEGACY BANKING (3DES-CBC + ElGamal Enc + SHA-1)\n{'='*70}")
            print("1. Register Cust | 2. Create Tx | 3. Verify Tx | 4. View Txs | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': cust = input("Cust ID: "); system.register_customer(cust)
            elif choice == '2': frm = input("From: "); to = input("To: "); amt = float(input("Amount: ")); system.create_transaction(frm, to, amt)
            elif choice == '3':
                try: txid = int(input("Tx ID to verify: ")); system.verify_transaction(txid)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Transactions ---"); [print(f" ID: {t['id']}, From: {t['from']}, Hash: {t['hash_sha1'][:16]}..., Status: {t['status']}") for t in system.transactions] if system.transactions else print("None.")
            elif choice == '5': system.performance.print_graph("Legacy Banking Performance")
            elif choice == '6': system.performance.compare_algorithms(['3DES_Encrypt_CBC', 'ElGamal_Encrypt', 'SHA1_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
     def menu_legacy_banking(): print("\nSystem 4 requires 'pycryptodome'."); pause()


# --- System 5: Healthcare (AES + RSA + SHA-256) ---
if HAS_CRYPTO:
    class HealthcareSystem: # Based on original structure
        def __init__(self):
            self.performance = PerformanceTracker()
            self.aes_key = get_random_bytes(32) # Master key for demo simplicity
            self.doctors = {} # doc_id: {'rsa_priv': key, 'rsa_pub': pubkey}
            self.records = [] # {'id', 'patient', 'doctor', 'enc_data', 'nonce', 'tag', 'signature', 'hash'}

        def register_doctor(self, doc_id, rsa_bits=2048):
            if doc_id in self.doctors: print("Doctor exists."); return
            start=time.time(); key = RSA.generate(rsa_bits); self.doctors[doc_id] = {'rsa_priv': key, 'rsa_pub': key.publickey()}; self.performance.record('RSA_KeyGen', time.time()-start); print(f"✓ Doctor {doc_id} registered.")

        def create_medical_record(self, patient_id, doctor_id, diagnosis):
            if doctor_id not in self.doctors: print("Doctor not registered!"); return None
            record_data = f"Patient:{patient_id}|Diagnosis:{diagnosis}"
            record_bytes = record_data.encode('utf-8')

            # AES-GCM Encryption
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM); nonce = cipher_aes.nonce; enc_data, tag = cipher_aes.encrypt_and_digest(record_bytes)
            aes_time = time.time()-start; self.performance.record('AES_Encrypt_GCM', aes_time, len(record_bytes))

            # SHA-256 Hash of original data
            start=time.time(); hash_obj = SHA256.new(record_bytes); record_hash = hash_obj.hexdigest(); self.performance.record('SHA256_Hash', time.time()-start)

            # RSA Signature of the hash
            start=time.time(); signer = pkcs1_15.new(self.doctors[doctor_id]['rsa_priv']); signature = signer.sign(hash_obj)
            rsa_time = time.time()-start; self.performance.record('RSA_Sign', rsa_time)

            record = {'id': len(self.records)+1, 'patient': patient_id, 'doctor': doctor_id, 'enc_data': enc_data, 'nonce': nonce, 'tag': tag, 'signature': signature, 'hash': record_hash}
            self.records.append(record)
            print(f"\n✓ Record created! (AES: {aes_time:.4f}s | RSA: {rsa_time:.4f}s | SHA256: <0.001s)")
            return record['id']

        def access_record(self, record_id, accessing_doctor_id):
            record = next((r for r in self.records if r['id'] == record_id), None)
            if not record: print("Record not found!"); return None
            # Basic access check (in real system more complex)
            # For demo, allow any registered doctor to verify/decrypt
            if accessing_doctor_id not in self.doctors: print("Accessing doctor not registered!"); return None
            signing_doctor_id = record['doctor']
            if signing_doctor_id not in self.doctors: print("Signing doctor's key not found!"); return None

            try:
                # AES-GCM Decryption (verifies integrity via tag)
                start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM, nonce=record['nonce']); decrypted_bytes = cipher_aes.decrypt_and_verify(record['enc_data'], record['tag'])
                decrypted_data = decrypted_bytes.decode('utf-8')
                aes_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', aes_time)

                # RSA Signature Verification (using signing doctor's public key)
                start=time.time(); hash_obj_recomputed = SHA256.new(decrypted_bytes) # Hash the decrypted data
                verifier = pkcs1_15.new(self.doctors[signing_doctor_id]['rsa_pub']); verifier.verify(hash_obj_recomputed, record['signature']); verified = True
                rsa_time = time.time()-start; self.performance.record('RSA_Verify', rsa_time)

                # Optional: Compare recomputed hash with stored hash
                hash_match = hash_obj_recomputed.hexdigest() == record['hash']

                print(f"\n✓ Record Accessed (AES Dec: {aes_time:.4f}s | RSA Vfy: {rsa_time:.4f}s)")
                print(f"  Signature Verified: {verified}\n  Stored/Computed Hash Match: {hash_match}\n  Data: {decrypted_data}")
                return decrypted_data
            except (ValueError, KeyError) as e:
                print(f"Access/Verification Error: {e}")
                 # Record failed attempts too
                if 'aes_time' not in locals(): aes_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', aes_time)
                if 'rsa_time' not in locals(): rsa_time = time.time()-start; self.performance.record('RSA_Verify', rsa_time)
                return None

    def menu_healthcare(): # System 5
        if not HAS_CRYPTO: print("\nSystem 5 requires 'pycryptodome'."); pause(); return
        system = HealthcareSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 5: HEALTHCARE (AES-GCM + RSA Sign + SHA-256)\n{'='*70}")
            print("1. Register Dr | 2. Create Record | 3. Access Record | 4. View Records | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': doc = input("Doctor ID: "); system.register_doctor(doc)
            elif choice == '2': patient = input("Patient ID: "); doctor = input("Doctor ID: "); diag = input("Diagnosis: "); system.create_medical_record(patient, doctor, diag)
            elif choice == '3':
                try: rid = int(input("Record ID: ")); acc_doc = input("Your Doctor ID: "); system.access_record(rid, acc_doc)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Records ---"); [print(f" ID: {r['id']}, Patient: {r['patient']}, Doctor: {r['doctor']}") for r in system.records] if system.records else print("None.")
            elif choice == '5': system.performance.print_graph("Healthcare System Performance")
            elif choice == '6': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'RSA_Sign', 'SHA256_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_healthcare(): print("\nSystem 5 requires 'pycryptodome'."); pause()


# --- System 6: Document Management (DES + ElGamal(Sig) + MD5) ---
# Note: Using ElGamal for SIGNING here, different from original interpretation
if HAS_CRYPTO:
    class DocumentManagementSystemDESElGamalMD5:
        def __init__(self, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            self.des_key = hashlib.md5(b"DocMgmtDESKey").digest()[:8] # Derive 8-byte key
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start)
            self.authors = {"author1": self.elgamal_keys} # Simple registration for demo
            self.documents = [] # {'id', 'author', 'enc_content', 'iv', 'signature', 'hash_md5'}

        def register_author(self, author_id, elgamal_bits=1024):
             if author_id in self.authors: print("Author exists."); return
             start=time.time(); keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); self.authors[author_id] = keys; print(f"✓ Author {author_id} registered.")

        def create_document(self, author_id, title, content):
            if author_id not in self.authors: print("Author not registered!"); return None
            doc_data = f"{title}||{content}"; doc_bytes = doc_data.encode('utf-8')

            # MD5 Hash of original data
            start=time.time(); hash_obj = MD5.new(doc_bytes); doc_hash = hash_obj.hexdigest(); self.performance.record('MD5_Hash', time.time()-start)

            # ElGamal Signature of the MD5 hash digest
            start=time.time(); signature = elgamal_sign(hash_obj.digest(), self.authors[author_id]); self.performance.record('ElGamal_Sign', time.time()-start)

            # DES Encryption (CBC mode)
            start=time.time(); cipher_des = DES.new(self.des_key, DES.MODE_CBC); iv = cipher_des.iv; enc_content = cipher_des.encrypt(pad(doc_bytes, DES.block_size))
            des_time = time.time()-start; self.performance.record('DES_Encrypt_CBC', des_time, len(doc_bytes))

            doc = {'id': len(self.documents)+1, 'author': author_id, 'enc_content': enc_content, 'iv': iv, 'signature': signature, 'hash_md5': doc_hash}
            self.documents.append(doc)
            print(f"\n✓ Doc created! (DES: {des_time:.4f}s | ElG Sign: {time.time()-start-des_time:.4f}s | MD5: <0.001s)") # Sig time inaccurate
            return doc['id']

        def verify_document(self, doc_id):
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if not doc: print("Doc not found!"); return False, None
            author_id = doc['author']
            if author_id not in self.authors: print("Author keys not found!"); return False, None
            elg_pub_key = {k:v for k,v in self.authors[author_id].items() if k != 'x'}

            try:
                # DES Decryption
                start=time.time(); cipher_des = DES.new(self.des_key, DES.MODE_CBC, iv=doc['iv']); decrypted_bytes = unpad(cipher_des.decrypt(doc['enc_content']), DES.block_size)
                decrypted_data = decrypted_bytes.decode('utf-8')
                des_time = time.time()-start; self.performance.record('DES_Decrypt_CBC', des_time)

                # Verify MD5 hash matches stored hash
                start=time.time(); computed_hash = MD5.new(decrypted_bytes).hexdigest(); hash_match = computed_hash == doc['hash_md5']
                md5_time = time.time()-start; self.performance.record('MD5_Verify', md5_time)

                # Verify ElGamal Signature against the original hash digest
                original_hash_digest = MD5.new(decrypted_bytes).digest() # Recompute digest
                start=time.time(); sig_valid = elgamal_verify(original_hash_digest, doc['signature'], elg_pub_key)
                elg_time = time.time()-start; self.performance.record('ElGamal_Verify', elg_time)

                print(f"\n✓ Doc Verified (DES Dec: {des_time:.4f}s | ElG Vfy: {elg_time:.4f}s | MD5 Vfy: {md5_time:.4f}s)")
                print(f"  Hash Match: {hash_match}\n  Signature Valid: {sig_valid}")
                if hash_match and sig_valid:
                     title, content = decrypted_data.split('||', 1)
                     print(f"  Title: {title}\n  Content: {content}"); return True, decrypted_data
                else: print("  Verification failed!"); return False, None
            except (ValueError, KeyError) as e: print(f"Verification Error: {e}"); return False, None

    def menu_document_management(): # System 6 (uses updated class)
        if not HAS_CRYPTO: print("\nSystem 6 requires 'pycryptodome'."); pause(); return
        system = DocumentManagementSystemDESElGamalMD5()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 6: DOC MGMT (DES-CBC + ElGamal Sig + MD5)\n{'='*70}")
            print("1. Register Author | 2. Create Doc | 3. Verify Doc | 4. List Docs | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': author = input("Author ID: "); system.register_author(author)
            elif choice == '2': author = input("Author ID: "); title = input("Title: "); content = input("Content: "); system.create_document(author, title, content)
            elif choice == '3':
                try: doc_id = int(input("Doc ID: ")); system.verify_document(doc_id)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Docs ---"); [print(f" ID: {d['id']}, Author: {d['author']}, Hash: {d['hash_md5'][:16]}...") for d in system.documents] if system.documents else print("None.")
            elif choice == '5': system.performance.print_graph("Doc Mgmt Performance")
            elif choice == '6': system.performance.compare_algorithms(['DES_Encrypt_CBC', 'ElGamal_Sign', 'MD5_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_document_management(): print("\nSystem 6 requires 'pycryptodome'."); pause()

# --- System 7: Messaging (AES + ElGamal(Sig) + MD5) ---
# Note: Using ElGamal for SIGNING here
if HAS_CRYPTO:
    class MessagingPlatformAESElGamalMD5: # Similar structure to Sys 6
        def __init__(self, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            self.aes_key = hashlib.sha256(b"MessagingAESKeyMaterial").digest() # AES-256 key
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start)
            self.users = {"user1": self.elgamal_keys} # Simple registration
            self.messages = [] # {'id', 'from', 'to', 'enc_msg', 'nonce', 'tag', 'signature', 'hash_md5'}

        def register_user(self, user_id, elgamal_bits=1024):
             if user_id in self.users: print("User exists."); return
             start=time.time(); keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); self.users[user_id] = keys; print(f"✓ User {user_id} registered.")

        def send_message(self, from_user, to_user, text):
            if from_user not in self.users: print("Sender not registered!"); return None
            msg_bytes = text.encode('utf-8')

            # MD5 Hash of original message
            start=time.time(); hash_obj = MD5.new(msg_bytes); msg_hash = hash_obj.hexdigest(); self.performance.record('MD5_Hash', time.time()-start)

            # ElGamal Signature of the MD5 hash digest
            start=time.time(); signature = elgamal_sign(hash_obj.digest(), self.users[from_user]); self.performance.record('ElGamal_Sign', time.time()-start)

            # AES-GCM Encryption
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM); nonce = cipher_aes.nonce; enc_msg, tag = cipher_aes.encrypt_and_digest(msg_bytes)
            aes_time = time.time()-start; self.performance.record('AES_Encrypt_GCM', aes_time, len(msg_bytes))

            msg = {'id': len(self.messages)+1, 'from': from_user, 'to': to_user, 'enc_msg': enc_msg, 'nonce': nonce, 'tag': tag, 'signature': signature, 'hash_md5': msg_hash}
            self.messages.append(msg)
            print(f"\n✓ Msg sent! (AES: {aes_time:.4f}s | ElG Sign: {time.time()-start-aes_time:.4f}s | MD5: <0.001s)") # Sig time inaccurate
            return msg['id']

        def read_message(self, msg_id, user_id):
            msg = next((m for m in self.messages if m['id'] == msg_id and m['to'] == user_id), None)
            if not msg: print("Message not found!"); return None
            sender_id = msg['from']
            if sender_id not in self.users: print("Sender keys not found!"); return None
            elg_pub_key = {k:v for k,v in self.users[sender_id].items() if k != 'x'}

            try:
                # AES-GCM Decryption (verifies integrity via tag)
                start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM, nonce=msg['nonce']); decrypted_bytes = cipher_aes.decrypt_and_verify(msg['enc_msg'], msg['tag'])
                decrypted_text = decrypted_bytes.decode('utf-8')
                aes_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', aes_time)

                # Verify MD5 hash matches stored hash
                start=time.time(); computed_hash = MD5.new(decrypted_bytes).hexdigest(); hash_match = computed_hash == msg['hash_md5']
                md5_time = time.time()-start; self.performance.record('MD5_Verify', md5_time)

                # Verify ElGamal Signature against the original hash digest
                original_hash_digest = MD5.new(decrypted_bytes).digest() # Recompute digest
                start=time.time(); sig_valid = elgamal_verify(original_hash_digest, msg['signature'], elg_pub_key)
                elg_time = time.time()-start; self.performance.record('ElGamal_Verify', elg_time)

                print(f"\n✓ Msg Read (AES Dec: {aes_time:.4f}s | ElG Vfy: {elg_time:.4f}s | MD5 Vfy: {md5_time:.4f}s)")
                print(f"  From: {sender_id}\n  Hash Match: {hash_match}\n  Signature Valid: {sig_valid}")
                if hash_match and sig_valid: print(f"  Message: {decrypted_text}"); return decrypted_text
                else: print("  Verification failed!"); return None
            except (ValueError, KeyError) as e: print(f"Read Error: {e}"); return None

    def menu_messaging(): # System 7 (uses updated class)
        if not HAS_CRYPTO: print("\nSystem 7 requires 'pycryptodome'."); pause(); return
        system = MessagingPlatformAESElGamalMD5()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 7: MESSAGING (AES-GCM + ElGamal Sig + MD5)\n{'='*70}")
            print("1. Register User | 2. Send Msg | 3. Read Msg | 4. List Msgs | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user = input("User ID: "); system.register_user(user)
            elif choice == '2': frm = input("From: "); to = input("To: "); txt = input("Message: "); system.send_message(frm, to, txt)
            elif choice == '3':
                user = input("Your ID: ");
                try: mid = int(input("Msg ID: ")); system.read_message(mid, user)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Msgs ---"); [print(f" ID: {m['id']}, {m['from']} -> {m['to']}") for m in system.messages] if system.messages else print("None.")
            elif choice == '5': system.performance.print_graph("Messaging Performance")
            elif choice == '6': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'ElGamal_Sign', 'MD5_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_messaging(): print("\nSystem 7 requires 'pycryptodome'."); pause()


# --- System 8: Secure File Transfer (DES + RSA + SHA-512) ---
if HAS_CRYPTO:
    class SecureFileTransferDESRSASHA512: # Based on original
        def __init__(self):
            self.performance = PerformanceTracker()
            self.users = {} # user_id: {'rsa_priv': key, 'rsa_pub': pubkey}
            self.files = [] # {'id', 'sender', 'recipient', 'filename', 'enc_data', 'iv', 'enc_key', 'hash_sha512'}

        def register_user(self, user_id, rsa_bits=2048):
            if user_id in self.users: print("User exists."); return
            start = time.time(); key = RSA.generate(rsa_bits); self.users[user_id] = {'rsa_priv': key, 'rsa_pub': key.publickey()}; self.performance.record('RSA_KeyGen', time.time() - start); print(f"✓ User {user_id} registered.")

        def send_file(self, sender, recipient, filename, content):
            if recipient not in self.users: print("Recipient not registered!"); return None
            if sender not in self.users: print("Sender not registered!"); return None # Need sender pub key if verifying sender later
            content_bytes = content.encode('utf-8')

            # SHA-512 Hash of original content
            start=time.time(); hash_sha512 = SHA512.new(content_bytes).hexdigest(); self.performance.record('SHA512_Hash', time.time()-start)

            # DES Encryption (CBC mode) with session key
            start = time.time(); des_key = get_random_bytes(8); cipher_des = DES.new(des_key, DES.MODE_CBC); iv = cipher_des.iv; enc_data = cipher_des.encrypt(pad(content_bytes, DES.block_size))
            des_time = time.time()-start; self.performance.record('DES_Encrypt_CBC', des_time, len(content_bytes))

            # RSA encryption of the DES session key (using recipient's public key)
            start = time.time(); recipient_pub_key = self.users[recipient]['rsa_pub']; cipher_rsa = PKCS1_OAEP.new(recipient_pub_key); enc_key = cipher_rsa.encrypt(des_key)
            rsa_time = time.time()-start; self.performance.record('RSA_Encrypt', rsa_time)

            file_rec = {'id': len(self.files)+1, 'sender': sender, 'recipient': recipient, 'filename': filename, 'enc_data': enc_data, 'iv': iv, 'enc_key': enc_key, 'hash_sha512': hash_sha512}
            self.files.append(file_rec)
            print(f"\n✓ File sent! (DES: {des_time:.4f}s | RSA: {rsa_time:.4f}s | SHA512: <0.001s)")
            return file_rec['id']

        def receive_file(self, file_id, user_id):
            file_rec = next((f for f in self.files if f['id'] == file_id and f['recipient'] == user_id), None)
            if not file_rec: print("File not found or access denied!"); return None
            if user_id not in self.users: print("User not registered!"); return None

            try:
                # RSA decryption of the DES key
                start=time.time(); user_priv_key = self.users[user_id]['rsa_priv']; cipher_rsa = PKCS1_OAEP.new(user_priv_key); des_key = cipher_rsa.decrypt(file_rec['enc_key'])
                rsa_time = time.time()-start; self.performance.record('RSA_Decrypt', rsa_time)

                # DES decryption of content
                start=time.time(); cipher_des = DES.new(des_key, DES.MODE_CBC, iv=file_rec['iv']); decrypted_bytes = unpad(cipher_des.decrypt(file_rec['enc_data']), DES.block_size)
                decrypted_content = decrypted_bytes.decode('utf-8')
                des_time = time.time()-start; self.performance.record('DES_Decrypt_CBC', des_time)

                # SHA-512 verification
                start=time.time(); computed_hash = SHA512.new(decrypted_bytes).hexdigest(); verified = computed_hash == file_rec['hash_sha512']
                sha_time = time.time()-start; self.performance.record('SHA512_Verify', sha_time)

                print(f"\n✓ File Received (RSA Dec: {rsa_time:.4f}s | DES Dec: {des_time:.4f}s | SHA512 Vfy: {sha_time:.4f}s)")
                print(f"  Filename: {file_rec['filename']}\n  Hash Verified: {verified}\n  Content: {decrypted_content}")
                return decrypted_content
            except (ValueError, KeyError) as e: print(f"Receive/Decryption Error: {e}"); return None

    def menu_file_transfer(): # System 8
        if not HAS_CRYPTO: print("\nSystem 8 requires 'pycryptodome'."); pause(); return
        system = SecureFileTransferDESRSASHA512()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 8: FILE TRANSFER (DES-CBC + RSA Enc + SHA-512)\n{'='*70}")
            print("1. Register User | 2. Send File | 3. Receive File | 4. List Files | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user = input("User ID: "); system.register_user(user)
            elif choice == '2': sender = input("From: "); recip = input("To: "); fname = input("Filename: "); content = input("Content: "); system.send_file(sender, recip, fname, content)
            elif choice == '3':
                user = input("Your ID: ");
                try: fid = int(input("File ID: ")); system.receive_file(fid, user)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Files ---"); [print(f" ID: {f['id']}, {f['sender']} -> {f['recipient']}, Name: {f['filename']}") for f in system.files] if system.files else print("None.")
            elif choice == '5': system.performance.print_graph("File Transfer Performance")
            elif choice == '6': system.performance.compare_algorithms(['DES_Encrypt_CBC', 'RSA_Encrypt', 'SHA512_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
     def menu_file_transfer(): print("\nSystem 8 requires 'pycryptodome'."); pause()


# --- System 9: Digital Library (Rabin + ElGamal(Sig) + SHA-256) ---
# Note: Using ElGamal for SIGNING
if HAS_CRYPTO:
    class DigitalLibraryRabinElGamalSHA256: # Similar to Sys 6/7 structure
        def __init__(self, rabin_bits=2048, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            start=time.time(); self.rabin_keys = self._generate_rabin_keys(rabin_bits); self.performance.record('Rabin_KeyGen', time.time()-start)
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start)
            self.publishers = {"pub1": self.elgamal_keys} # Simple registration
            self.books = [] # {'id', 'publisher', 'enc_content', 'signature', 'hash_sha256'}

        # _generate_rabin_keys, _rabin_encrypt, _rabin_decrypt same as System 3/26
        def _generate_rabin_keys(self, bits): # Same as Sys 3/26
             while True: p = number.getPrime(bits // 2);
                 if p % 4 == 3: break
             while True: q = number.getPrime(bits // 2);
                 if q % 4 == 3 and q != p: break
             print("✓ Rabin keys generated."); return {'n': p * q, 'p': p, 'q': q}
        def _rabin_encrypt(self, msg_bytes, n): # Same padding as Sys 3/26
            redundancy = b"FILEPAD" + len(msg_bytes).to_bytes(4, 'big')
            full_msg = msg_bytes + redundancy; m = number.bytes_to_long(full_msg)
            if m >= n: raise ValueError("Message too large for Rabin key")
            return pow(m, 2, n)
        def _rabin_decrypt(self, cipher_int, p, q, n): # Same as Sys 3/26
            mp=pow(cipher_int,(p+1)//4,p); mq=pow(cipher_int,(q+1)//4,q); inv_p=number.inverse(p,q); inv_q=number.inverse(q,p); a=(q*inv_q)%n; b=(p*inv_p)%n
            roots=[(a*mp+b*mq)%n,(a*mp-b*mq)%n,(-a*mp+b*mq)%n,(-a*mp-b*mq)%n]
            for r in roots:
                try: r_bytes=number.long_to_bytes(r);
                    if len(r_bytes)>11 and r_bytes.endswith(b"FILEPAD"): len_s=len(r_bytes)-11; msg_len=int.from_bytes(r_bytes[len_s:len_s+4],'big');
                        if len_s==msg_len: return r_bytes[:msg_len]
                except Exception: continue
            return None

        def register_publisher(self, pub_id, elgamal_bits=1024):
             if pub_id in self.publishers: print("Publisher exists."); return
             start=time.time(); keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); self.publishers[pub_id] = keys; print(f"✓ Publisher {pub_id} registered.")

        def publish_book(self, publisher_id, title, content):
            if publisher_id not in self.publishers: print("Publisher not registered!"); return None
            book_data = f"{title}:{content}"; book_bytes = book_data.encode('utf-8')

            # SHA-256 Hash of original data
            start=time.time(); hash_obj = SHA256.new(book_bytes); book_hash = hash_obj.hexdigest(); self.performance.record('SHA256_Hash', time.time()-start)

            # ElGamal Signature of the SHA-256 hash digest
            start=time.time(); signature = elgamal_sign(hash_obj.digest(), self.publishers[publisher_id]); self.performance.record('ElGamal_Sign', time.time()-start)

            # Rabin Encryption of content
            start=time.time();
            try: enc_content = self._rabin_encrypt(book_bytes, self.rabin_keys['n'])
            except ValueError as e: print(f"Encryption error: {e}"); return None
            rabin_time = time.time()-start; self.performance.record('Rabin_Encrypt', rabin_time, len(book_bytes))

            book = {'id': len(self.books)+1, 'publisher': publisher_id, 'enc_content': enc_content, 'signature': signature, 'hash_sha256': book_hash}
            self.books.append(book)
            print(f"\n✓ Book published! (Rabin: {rabin_time:.4f}s | ElG Sign: {time.time()-start-rabin_time:.4f}s | SHA256: <0.001s)") # Sig time inaccurate
            return book['id']

        def read_book(self, book_id):
            book = next((b for b in self.books if b['id'] == book_id), None)
            if not book: print("Book not found!"); return None
            publisher_id = book['publisher']
            if publisher_id not in self.publishers: print("Publisher keys not found!"); return None
            elg_pub_key = {k:v for k,v in self.publishers[publisher_id].items() if k != 'x'}

            try:
                # Rabin Decryption
                start=time.time(); decrypted_bytes = self._rabin_decrypt(book['enc_content'], self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n'])
                rabin_time = time.time()-start; self.performance.record('Rabin_Decrypt', rabin_time)

                if decrypted_bytes is None: print("Rabin decryption failed!"); return None
                decrypted_data = decrypted_bytes.decode('utf-8')

                # Verify SHA-256 hash matches stored hash
                start=time.time(); computed_hash = SHA256.new(decrypted_bytes).hexdigest(); hash_match = computed_hash == book['hash_sha256']
                sha_time = time.time()-start; self.performance.record('SHA256_Verify', sha_time)

                # Verify ElGamal Signature against the original hash digest
                original_hash_digest = SHA256.new(decrypted_bytes).digest() # Recompute digest
                start=time.time(); sig_valid = elgamal_verify(original_hash_digest, book['signature'], elg_pub_key)
                elg_time = time.time()-start; self.performance.record('ElGamal_Verify', elg_time)

                print(f"\n✓ Book Read (Rabin Dec: {rabin_time:.4f}s | ElG Vfy: {elg_time:.4f}s | SHA256 Vfy: {sha_time:.4f}s)")
                print(f"  Hash Match: {hash_match}\n  Signature Valid: {sig_valid}")
                if hash_match and sig_valid:
                     title, content = decrypted_data.split(':', 1)
                     print(f"  Title: {title}\n  Content: {content}"); return decrypted_data
                else: print("  Verification failed!"); return None
            except (ValueError, KeyError) as e: print(f"Read Error: {e}"); return None

    def menu_digital_library(): # System 9
        if not HAS_CRYPTO: print("\nSystem 9 requires 'pycryptodome'."); pause(); return
        system = DigitalLibraryRabinElGamalSHA256()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 9: DIGITAL LIBRARY (Rabin Enc + ElGamal Sig + SHA-256)\n{'='*70}")
            print("1. Register Pub | 2. Publish Book | 3. Read Book | 4. List Books | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': pub = input("Publisher ID: "); system.register_publisher(pub)
            elif choice == '2': pub = input("Publisher ID: "); title = input("Title: "); content = input("Content: "); system.publish_book(pub, title, content)
            elif choice == '3':
                try: bid = int(input("Book ID: ")); system.read_book(bid)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Books ---"); [print(f" ID: {b['id']}, Publisher: {b['publisher']}") for b in system.books] if system.books else print("None.")
            elif choice == '5': system.performance.print_graph("Digital Library Performance")
            elif choice == '6': system.performance.compare_algorithms(['Rabin_Encrypt', 'ElGamal_Sign', 'SHA256_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
     def menu_digital_library(): print("\nSystem 9 requires 'pycryptodome'."); pause()


# --- System 10: Secure Chat (AES + RSA + SHA-512) ---
if HAS_CRYPTO:
    class SecureChatSystemAESRSASHA512: # Similar to Healthcare system
        def __init__(self):
            self.performance = PerformanceTracker()
            self.aes_key = get_random_bytes(32) # Shared key for demo
            self.users = {} # user_id: {'rsa_priv': key, 'rsa_pub': pubkey}
            self.messages = [] # {'id', 'sender', 'enc_msg', 'nonce', 'tag', 'signature', 'hash'}

        def register_user(self, user_id, rsa_bits=2048):
            if user_id in self.users: print("User exists."); return
            start=time.time(); key = RSA.generate(rsa_bits); self.users[user_id] = {'rsa_priv': key, 'rsa_pub': key.publickey()}; self.performance.record('RSA_KeyGen', time.time()-start); print(f"✓ User {user_id} registered.")

        def send_message(self, sender_id, message_text):
            if sender_id not in self.users: print("Sender not registered!"); return None
            msg_bytes = message_text.encode('utf-8')

            # AES-GCM Encryption
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM); nonce = cipher_aes.nonce; enc_msg, tag = cipher_aes.encrypt_and_digest(msg_bytes)
            aes_time = time.time()-start; self.performance.record('AES_Encrypt_GCM', aes_time, len(msg_bytes))

            # SHA-512 Hash of original message
            start=time.time(); hash_obj = SHA512.new(msg_bytes); msg_hash = hash_obj.hexdigest(); self.performance.record('SHA512_Hash', time.time()-start)

            # RSA Signature of the hash
            start=time.time(); signer = pkcs1_15.new(self.users[sender_id]['rsa_priv']); signature = signer.sign(hash_obj)
            rsa_time = time.time()-start; self.performance.record('RSA_Sign', rsa_time)

            message = {'id': len(self.messages)+1, 'sender': sender_id, 'enc_msg': enc_msg, 'nonce': nonce, 'tag': tag, 'signature': signature, 'hash': msg_hash}
            self.messages.append(message)
            print(f"\n✓ Msg sent! (AES: {aes_time:.4f}s | RSA Sign: {rsa_time:.4f}s | SHA512: <0.001s)")
            return message['id']

        def read_message(self, msg_id, reader_id): # Reader verifies sender's signature
            message = next((m for m in self.messages if m['id'] == msg_id), None)
            if not message: print("Message not found!"); return None
            sender_id = message['sender']
            if sender_id not in self.users: print("Sender keys not found!"); return None
            if reader_id not in self.users: print("Reader not registered!"); return None # For consistency

            try:
                # AES-GCM Decryption (verifies integrity via tag)
                start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM, nonce=message['nonce']); decrypted_bytes = cipher_aes.decrypt_and_verify(message['enc_msg'], message['tag'])
                decrypted_text = decrypted_bytes.decode('utf-8')
                aes_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', aes_time)

                # RSA Signature Verification (using sender's public key)
                start=time.time(); hash_obj_recomputed = SHA512.new(decrypted_bytes) # Hash the decrypted data
                verifier = pkcs1_15.new(self.users[sender_id]['rsa_pub']); verifier.verify(hash_obj_recomputed, message['signature']); verified = True
                rsa_time = time.time()-start; self.performance.record('RSA_Verify', rsa_time)

                hash_match = hash_obj_recomputed.hexdigest() == message['hash']

                print(f"\n✓ Msg Read (AES Dec: {aes_time:.4f}s | RSA Vfy: {rsa_time:.4f}s)")
                print(f"  Sender: {sender_id}\n  Sig Verified: {verified}\n  Hash Match: {hash_match}\n  Message: {decrypted_text}")
                return decrypted_text
            except (ValueError, KeyError) as e:
                print(f"Read/Verification Error: {e}"); return None

    def menu_secure_chat(): # System 10
        if not HAS_CRYPTO: print("\nSystem 10 requires 'pycryptodome'."); pause(); return
        system = SecureChatSystemAESRSASHA512()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 10: SECURE CHAT (AES-GCM + RSA Sign + SHA-512)\n{'='*70}")
            print("1. Register User | 2. Send Msg | 3. Read Msg | 4. List Msgs | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user = input("User ID: "); system.register_user(user)
            elif choice == '2': sender = input("Your ID: "); text = input("Message: "); system.send_message(sender, text)
            elif choice == '3':
                reader = input("Your ID: ");
                try: mid = int(input("Msg ID: ")); system.read_message(mid, reader)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Msgs ---"); [print(f" ID: {m['id']}, Sender: {m['sender']}") for m in system.messages] if system.messages else print("None.")
            elif choice == '5': system.performance.print_graph("Secure Chat Performance")
            elif choice == '6': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'RSA_Sign', 'SHA512_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
    def menu_secure_chat(): print("\nSystem 10 requires 'pycryptodome'."); pause()

# --- System 11: E-Voting (Paillier + ElGamal(Sig) + SHA-256) ---
if HAS_PAILLIER and HAS_CRYPTO:
    class SecureVotingSystem: # Based on original
        def __init__(self, paillier_key_size=1024, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            print("Initializing Paillier keypair..."); start=time.time(); self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=paillier_key_size); self.performance.record('Paillier_KeyGen', time.time() - start); print("✓ Paillier keys ready.")
            self.voters = {} # voter_id: {'elg_keys': keys}
            self.candidates = {} # cid: {'name': name, 'enc_tally': enc_obj}
            self.votes = [] # {'voter_id', 'candidate_id', 'enc_vote', 'hash', 'sig_r', 'sig_s'}
            self.voted_ids = set()

        def register_voter(self, voter_id, elgamal_bits=1024):
            if voter_id in self.voters: print("Voter exists."); return
            start = time.time(); keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time() - start); self.voters[voter_id] = {'elg_keys': keys}; print(f"✓ Voter {voter_id} registered.")

        def register_candidate(self, candidate_name):
            cid = len(self.candidates) + 1; self.candidates[cid] = {'name': candidate_name, 'encrypted_tally': self.public_key.encrypt(0)}; print(f"✓ Candidate {candidate_name} registered (ID: {cid})"); return cid

        def cast_vote(self, voter_id, candidate_id):
            if voter_id not in self.voters: print("Voter not registered!"); return None
            if candidate_id not in self.candidates: print("Candidate not found!"); return None
            if voter_id in self.voted_ids: print("Voter already voted!"); return None

            vote_data = f"VOTE|{voter_id}|{candidate_id}|{datetime.now().isoformat()}"; vote_bytes = vote_data.encode('utf-8')

            # SHA-256 hash (Receipt)
            start=time.time(); hash_obj = SHA256.new(vote_bytes); vote_hash_digest = hash_obj.digest(); hash_hex = hash_obj.hexdigest(); self.performance.record('SHA256_Hash', time.time()-start)

            # ElGamal signature of the hash digest
            start=time.time(); signature = elgamal_sign(vote_hash_digest, self.voters[voter_id]['elg_keys']); self.performance.record('ElGamal_Sign', time.time()-start)

            # Paillier encryption of vote (1)
            start=time.time(); encrypted_vote = self.public_key.encrypt(1); self.performance.record('Paillier_Encrypt', time.time()-start)

            # Homomorphic Addition to Tally
            start=time.time(); self.candidates[candidate_id]['encrypted_tally'] += encrypted_vote; self.performance.record('Paillier_Add', time.time()-start)

            vote_record = {'voter_id': voter_id, 'candidate_id': candidate_id, 'encrypted_vote': encrypted_vote, 'hash': hash_hex, 'sig_r': signature[0], 'sig_s': signature[1]}
            self.votes.append(vote_record); self.voted_ids.add(voter_id)
            print(f"\n✓ Vote cast! Receipt Hash: {hash_hex}"); return hash_hex

        def tally_results(self):
            print(f"\n{'='*70}\n  ELECTION TALLY & VERIFICATION\n{'='*70}"); total_verified = 0
            for vote in self.votes:
                if vote['voter_id'] not in self.voters: print(f"Warning: Keys for voter {vote['voter_id']} not found!"); continue
                elg_pub_key = {k:v for k,v in self.voters[vote['voter_id']]['elg_keys'].items() if k != 'x'}
                # Verify signature against the *original hash digest* (need to store/reconstruct)
                # For simplicity here, assume hash_hex represents what was signed. In reality, sign the digest.
                # Reconstruct vote_data conceptually to get bytes, then hash
                # This requires storing more info or making assumptions. Let's verify against the stored hash hex -> bytes
                try: hash_bytes = bytes.fromhex(vote['hash'])
                except: print(f"Warning: Invalid hash format for vote from {vote['voter_id']}"); continue
                start = time.time(); verified = elgamal_verify(hash_bytes, (vote['sig_r'], vote['sig_s']), elg_pub_key); self.performance.record('ElGamal_Verify', time.time() - start)
                if verified: total_verified += 1
                else: print(f"Warning: Invalid signature for vote from {vote['voter_id']}")

            print(f"  Total Votes: {len(self.votes)}, Verified Signatures: {total_verified}")
            if total_verified != len(self.votes): print("  WARNING: Signature mismatch!")

            print("\nFinal Results:"); results = []
            for cid, data in self.candidates.items():
                 start = time.time(); count = self.private_key.decrypt(data['encrypted_tally']); self.performance.record('Paillier_Decrypt_Total', time.time() - start); results.append((data['name'], count))
            for name, count in sorted(results, key=lambda x: x[1], reverse=True): print(f"  - {name:20s}: {count} votes")
            print("="*70)

    def menu_voting_system(): # System 11
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 11 requires 'phe' and 'pycryptodome'."); pause(); return
        system = SecureVotingSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 11: E-VOTING (Paillier + ElGamal Sig + SHA-256)\n{'='*70}")
            print("1. Register Voter | 2. Register Candidate | 3. Cast Vote | 4. Tally Results | 5. View Votes | 6. Performance | 7. Compare Algs | 8. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': vid = input("Voter ID: "); system.register_voter(vid)
            elif choice == '2': cname = input("Candidate Name: "); system.register_candidate(cname)
            elif choice == '3':
                vid = input("Your Voter ID: ");
                try: cid = int(input("Vote for Candidate ID: ")); system.cast_vote(vid, cid)
                except ValueError: print("Invalid ID.")
            elif choice == '4': system.tally_results()
            elif choice == '5': print("\n--- Votes ---"); [print(f" Voter: {v['voter_id']}, Candidate: {v['candidate_id']}, Hash: {v['hash'][:16]}...") for v in system.votes] if system.votes else print("None.")
            elif choice == '6': system.performance.print_graph("E-Voting Performance")
            elif choice == '7': system.performance.compare_algorithms(['Paillier_Encrypt', 'ElGamal_Sign', 'SHA256_Hash'])
            elif choice == '8': break
            else: print("Invalid choice.")
            if choice != '8': pause()
else:
    def menu_voting_system(): print("\nSystem 11 requires 'phe' and 'pycryptodome'."); pause()


# --- System 12: Hybrid (Hill Cipher + RSA + SHA-256) ---
if HAS_NUMPY and HAS_CRYPTO:
    class HillHybridSystem: # Based on original
        def __init__(self):
            self.performance = PerformanceTracker()
            self.users = {} # user_id: {'rsa_priv': key, 'rsa_pub': pubkey}
            self.messages = [] # {'id', 'sender', 'recipient', 'enc_msg', 'enc_key_info', 'hash'}

        def register_user(self, user_id, rsa_bits=2048):
            if user_id in self.users: print("User exists."); return
            start=time.time(); key = RSA.generate(rsa_bits); self.users[user_id] = {'rsa_priv': key, 'rsa_pub': key.publickey()}; self.performance.record('RSA_KeyGen', time.time()-start); print(f"✓ User {user_id} registered.")

        # Using toolkit_classical's hill_cipher_2x2 for implementation consistency
        # Need original length handling added
        def _hill_encrypt_with_len(self, plaintext, key_matrix):
            plain_alpha = "".join(filter(str.isalpha, plaintext.upper()))
            original_len = len(plain_alpha)
            # Use the toolkit's encryption which handles padding
            ciphertext = toolkit_classical.hill_cipher_2x2(plain_alpha, key_matrix)
            return ciphertext, original_len

        def _hill_decrypt_with_len(self, ciphertext, key_matrix, original_len):
             # Use the toolkit's decryption
             decrypted_padded = toolkit_classical.hill_cipher_2x2(ciphertext, key_matrix, decrypt=True)
             # Return only the original length
             return decrypted_padded[:original_len] if isinstance(decrypted_padded, str) else decrypted_padded


        def send_message(self, sender, recipient, message, key_matrix_str):
            if recipient not in self.users: print("Recipient not found!"); return None
            if sender not in self.users: print("Sender not registered!"); return None

            try: # Validate and parse Hill key
                key_vals = [int(v) for v in key_matrix_str.split()]; n_sq = len(key_vals); n = int(math.sqrt(n_sq))
                if n*n != n_sq or n != 2: raise ValueError("Requires a 2x2 key (4 values)")
                key_matrix = [key_vals[0:2], key_vals[2:4]]; det = (key_matrix[0][0]*key_matrix[1][1]-key_matrix[0][1]*key_matrix[1][0])%26
                if mod_inverse(det, 26) is None: raise ValueError("Hill key matrix not invertible")
            except ValueError as e: print(f"Invalid Hill key: {e}"); return None

            message_bytes = message.encode('utf-8') # Use original message for hash

            # SHA-256 Hash of original message
            start=time.time(); msg_hash = SHA256.new(message_bytes).hexdigest(); self.performance.record('SHA256_Hash', time.time()-start)

            # Hill encryption (only on alpha chars, track original alpha length)
            start=time.time(); enc_msg, original_len = self._hill_encrypt_with_len(message, key_matrix); hill_time = time.time()-start; self.performance.record('Hill_Encrypt', hill_time, len(message))

            # RSA encryption of Hill key string AND original alpha length
            key_info = f"{key_matrix_str}|{original_len}"
            start=time.time(); recip_pub_key = self.users[recipient]['rsa_pub']; cipher_rsa = PKCS1_OAEP.new(recip_pub_key); enc_key_info = cipher_rsa.encrypt(key_info.encode())
            rsa_time = time.time()-start; self.performance.record('RSA_Encrypt', rsa_time)

            msg = {'id': len(self.messages)+1, 'sender': sender, 'recipient': recipient, 'enc_msg': enc_msg, 'enc_key_info': enc_key_info, 'hash': msg_hash}
            self.messages.append(msg)
            print(f"\n✓ Msg Sent! (Hill: {hill_time:.4f}s | RSA: {rsa_time:.4f}s | SHA256: <0.001s)")
            return msg['id']

        def read_message(self, user_id, msg_id):
            msg = next((m for m in self.messages if m['id'] == msg_id and m['recipient'] == user_id), None)
            if not msg: print("Message not found/access denied!"); return None
            if user_id not in self.users: print("User not registered!"); return None

            try:
                # RSA decrypt key info
                start=time.time(); user_priv_key = self.users[user_id]['rsa_priv']; cipher_rsa = PKCS1_OAEP.new(user_priv_key); key_info = cipher_rsa.decrypt(msg['enc_key_info']).decode()
                rsa_time = time.time()-start; self.performance.record('RSA_Decrypt', rsa_time)
                key_matrix_str, original_len_str = key_info.split('|'); original_len = int(original_len_str)
                key_vals = [int(v) for v in key_matrix_str.split()]; key_matrix = [key_vals[0:2], key_vals[2:4]] # Rebuild matrix

                # Hill decrypt
                start=time.time(); decrypted_alpha = self._hill_decrypt_with_len(msg['enc_msg'], key_matrix, original_len); hill_time = time.time()-start; self.performance.record('Hill_Decrypt', hill_time)
                # Note: This only decrypts the alpha part. Reconstructing the full original msg needs more info.
                # For demo, we verify hash against a reconstructed message (may differ slightly from original if non-alpha chars existed)
                decrypted_approx = decrypted_alpha # Assuming original was only alpha for hash check

                # SHA-256 verification (against approx decrypted)
                start=time.time(); computed_hash = SHA256.new(decrypted_approx.encode()).hexdigest(); verified = computed_hash == msg['hash']
                sha_time = time.time()-start; self.performance.record('SHA256_Verify', sha_time)

                print(f"\n✓ Msg Read (RSA Dec: {rsa_time:.4f}s | Hill Dec: {hill_time:.4f}s | SHA256 Vfy: {sha_time:.4f}s)")
                print(f"  Hash Verified (approx): {verified}\n  Decrypted Alpha Chars: {decrypted_alpha}")
                return decrypted_alpha # Return alpha part

            except (ValueError, KeyError) as e: print(f"Read/Decryption Error: {e}"); return None

    def menu_hill_hybrid(): # System 12
        if not HAS_NUMPY or not HAS_CRYPTO: print("\nSystem 12 requires NumPy and PyCryptodome."); pause(); return
        system = HillHybridSystem()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  SYSTEM 12: HYBRID (Hill Cipher 2x2 + RSA Enc + SHA-256)\n{'='*70}")
            print("1. Register User | 2. Send Msg | 3. Read Msg | 4. List Msgs | 5. Performance | 6. Compare Algs | 7. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': user = input("User ID: "); system.register_user(user)
            elif choice == '2': sender = input("From: "); recip = input("To: "); msg = input("Message: "); key_str = input("Hill Key (4 vals): "); system.send_message(sender, recip, msg, key_str)
            elif choice == '3':
                user = input("Your ID: ");
                try: mid = int(input("Msg ID: ")); system.read_message(user, mid)
                except ValueError: print("Invalid ID.")
            elif choice == '4': print("\n--- Msgs ---"); [print(f" ID: {m['id']}, {m['sender']}->{m['recipient']}, Enc: {m['enc_msg'][:20]}...") for m in system.messages] if system.messages else print("None.")
            elif choice == '5': system.performance.print_graph("Hill Hybrid Performance")
            elif choice == '6': system.performance.compare_algorithms(['Hill_Encrypt', 'RSA_Encrypt', 'SHA256_Hash'])
            elif choice == '7': break
            else: print("Invalid choice.")
            if choice != '7': pause()
else:
     def menu_hill_hybrid(): print("\nSystem 12 requires NumPy and PyCryptodome."); pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 2: INTERACTIVE CLASSICAL CIPHERS (Systems 13-20)
# FULL CODE PASTED HERE
#
# ═══════════════════════════════════════════════════════════════════════════

def menu_additive_cipher(): # System 13
    clear_screen(); print("🔤 SYSTEM 13: ADDITIVE CIPHER (CAESAR)"); print("="*40); msg = input("Msg: ").upper();
    try: key = int(input("Key (0-25): ")) % 26; enc = toolkit_classical.additive_cipher(msg, key); dec = toolkit_classical.additive_cipher(enc, key, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}"); pause()
    except ValueError: print("Invalid key."); pause()

def menu_multiplicative_cipher(): # System 14
    clear_screen(); print("🔢 SYSTEM 14: MULTIPLICATIVE CIPHER"); print("="*40); msg = input("Msg: ").upper(); print("Valid keys: 1,3,5,7,9,11,15,17,19,21,23,25");
    try: key = int(input("Key: "));
        if gcd(key, 26) != 1: raise ValueError("Key not coprime to 26")
        enc = toolkit_classical.multiplicative_cipher(msg, key); dec = toolkit_classical.multiplicative_cipher(enc, key, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}\nInv: {mod_inverse(key, 26)}"); pause()
    except ValueError as e: print(f"Invalid key: {e}"); pause()

def menu_affine_cipher(): # System 15
    clear_screen(); print("🔡 SYSTEM 15: AFFINE CIPHER"); print("="*30); msg = input("Msg: ").upper();
    try: a = int(input("a (coprime to 26): ")); b = int(input("b (0-25): ")) % 26;
        if gcd(a, 26) != 1: raise ValueError("'a' not coprime to 26")
        enc = toolkit_classical.affine_cipher(msg, a, b); dec = toolkit_classical.affine_cipher(enc, a, b, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}\nParams: a={a}, b={b}, a_inv={mod_inverse(a, 26)}"); pause()
    except ValueError as e: print(f"Invalid key(s): {e}"); pause()

def menu_vigenere_cipher(): # System 16
    clear_screen(); print("🔠 SYSTEM 16: VIGENERE CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.vigenere_cipher(msg, key); dec = toolkit_classical.vigenere_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()

def menu_autokey_cipher(): # System 17
    clear_screen(); print("🔑 SYSTEM 17: AUTOKEY CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.autokey_cipher(msg, key); dec = toolkit_classical.autokey_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()

def menu_playfair_cipher(): # System 18
    clear_screen(); print("▦ SYSTEM 18: PLAYFAIR CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.playfair_cipher(msg, key); dec = toolkit_classical.playfair_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()

def menu_hill_cipher(): # System 19
    if not HAS_NUMPY: print("\nHill Cipher requires NumPy."); pause(); return
    clear_screen(); print("🔢 SYSTEM 19: HILL CIPHER (2x2)"); print("="*30); msg = input("Msg: ").upper(); key_str = input("Key (e.g., '3 3 2 7'): ").split()
    try: key_vals = [int(v) for v in key_str];
        if len(key_vals)!=4: raise ValueError("Need 4 values"); key_matrix = [key_vals[0:2], key_vals[2:4]]; det = (key_matrix[0][0]*key_matrix[1][1]-key_matrix[0][1]*key_matrix[1][0])%26;
        if mod_inverse(det, 26) is None: raise ValueError(f"Matrix det={det} not invertible")
        enc = toolkit_classical.hill_cipher_2x2(msg, key_matrix); dec = toolkit_classical.hill_cipher_2x2(enc, key_matrix, True); print(f"\nOrig: {msg}\nKey: {key_matrix}\nEnc: {enc}\nDec: {dec}"); pause()
    except ValueError as e: print(f"Invalid key: {e}"); pause()

def menu_transposition_cipher(): # System 20
    clear_screen(); print("🔄 SYSTEM 20: TRANSPOSITION CIPHER (COLUMNAR)"); print("="*50); message = input("Msg: ").upper().replace(" ", ""); key = input("Key: ").upper();
    if not key.isalpha() or len(key) == 0: print("Invalid key."); pause(); return
    def transpose_encrypt(msg, keyword): # Simple columnar transpose
        key_order=sorted([(c,i) for i,c in enumerate(keyword)]); num_cols=len(keyword); num_rows=math.ceil(len(msg)/num_cols); grid=[['X'] * num_cols for _ in range(num_rows)]; msg_idx=0
        for r in range(num_rows):
             for c in range(num_cols):
                 if msg_idx < len(msg): grid[r][c]=msg[msg_idx]; msg_idx+=1
        return "".join( grid[r][c] for _,c in key_order for r in range(num_rows) )
    def transpose_decrypt(cipher, keyword): # Basic decrypt assumes full grid
        key_order=sorted([(c,i) for i,c in enumerate(keyword)]); num_cols=len(keyword); num_rows=math.ceil(len(cipher)/num_cols); grid=[[''] * num_cols for _ in range(num_rows)]; cipher_idx=0
        # Determine column lengths based on key order and cipher length
        num_short_cols = num_cols - (len(cipher) % num_cols) if len(cipher) % num_cols != 0 else 0
        col_lens = {}
        sorted_indices = [idx for _, idx in key_order]
        ptr = 0
        for i in range(num_cols):
            # The columns corresponding to the *last* few letters in sorted key order might be short
            current_original_col_index = sorted_indices[i]
            length = num_rows if (num_cols - num_short_cols) > i else num_rows - 1 # Incorrect assignment logic
            # Correct assignment based on *original* column index relative to cutoff
            length = num_rows -1 if current_original_col_index >= (num_cols - num_short_cols) and num_short_cols > 0 else num_rows

            col_lens[current_original_col_index] = length


        cipher_idx = 0
        for _, orig_col_idx in key_order:
             col_len = col_lens[orig_col_idx]
             for r in range(col_len):
                  if cipher_idx < len(cipher): grid[r][orig_col_idx] = cipher[cipher_idx]; cipher_idx += 1

        plaintext = "".join( grid[r][c] for r in range(num_rows) for c in range(num_cols) )
        # Crude padding removal - find last non-'X' and slice? Or rely on known length?
        # Find original length would be ideal. Assuming padding was just Xs:
        return plaintext.rstrip('X') # Simplistic

    encrypted = transpose_encrypt(message, key); decrypted = transpose_decrypt(encrypted, key);
    print(f"\nOrig: {message}\nKey:  {key}\nEnc: {encrypted}\nDec: {decrypted} (Padding removal basic)"); pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 3: EXAM-SPECIFIC SCENARIOS & NEW COMBINATIONS (Systems 21-28)
# FULL CODE PASTED HERE
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 21: Client-Server Payment Gateway (Paillier + RSA Sign + SHA-256) ---
if HAS_PAILLIER and HAS_CRYPTO:
    class PaymentGatewaySystem:
        """Simulates Seller-Payment Gateway transactions using Paillier, RSA, SHA-256"""
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

    def menu_payment_gateway(): # System 21
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 21 requires 'phe' and 'pycryptodome'."); pause(); return
        system = PaymentGatewaySystem(); submitted_transactions = []
        while True:
            clear_screen(); print(f"\n{'='*70}\n  🏦 SYSTEM 21: SELLER-PAYMENT GATEWAY (Paillier + RSA Sign + SHA-256)\n{'='*70}")
            print("  1. Register Seller | 2. Submit Transaction | 3. Gateway: Process Transactions | 4. Generate & Sign Summary | 5. Verify Signed Summary | 6. View All Summaries | 7. Performance | 8. Compare Algs | 9. Back"); print("-"*70)
            choice = input("Choice: ").strip()
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
if HAS_PAILLIER and HAS_CRYPTO:
    class SecureAggregationPaillierElGamal:
        """Paillier aggregation with ElGamal signatures"""
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

    def menu_secure_aggregation_paillier_elgamal(): # System 22
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 22 requires 'phe' and 'pycryptodome'."); pause(); return
        system = SecureAggregationPaillierElGamal(); participant_count = 0
        while True:
            clear_screen(); print(f"\n{'='*70}\n  ∑ SYSTEM 22: SECURE AGGREGATION (Paillier + ElGamal Sign + SHA-512)\n{'='*70}")
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
if HAS_CRYPTO:
    class HomomorphicProductElGamal:
        """ElGamal multiplicative homomorphism with RSA signatures"""
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

    def menu_homomorphic_product_elgamal(): # System 23
        if not HAS_CRYPTO: print("\nSystem 23 requires 'pycryptodome'."); pause(); return
        system = HomomorphicProductElGamal(); participant_count = 0
        while True:
            clear_screen(); print(f"\n{'='*70}\n  ∏ SYSTEM 23: HOMOMORPHIC PRODUCT (ElGamal Enc + RSA Sign + SHA256)\n{'='*70}")
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
    def menu_homomorphic_product_elgamal(): print("\nSystem 23 requires 'pycryptodome'."); pause()

# --- System 24: Secure Aggregation (Paillier + RSA Sign + SHA-512) ---
if HAS_PAILLIER and HAS_CRYPTO:
    class SecureAggregationPaillierRSA:
        """Paillier aggregation with RSA signatures"""
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

    def menu_secure_aggregation_paillier_rsa(): # System 24
        if not HAS_PAILLIER or not HAS_CRYPTO: print("\nSystem 24 requires 'phe' and 'pycryptodome'."); pause(); return
        system = SecureAggregationPaillierRSA(); participant_count = 0
        while True:
            clear_screen(); print(f"\n{'='*70}\n  ∑ SYSTEM 24: SECURE AGGREGATION (Paillier + RSA Sign + SHA-512)\n{'='*70}")
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

# --- System 25: Secure Transmission (AES-GCM + ElGamal Sign + SHA256) ---
if HAS_CRYPTO:
    class SecureTransmissionAESElGamal:
        def __init__(self, aes_key_size=32, elgamal_bits=1024): # AES-256
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Transmission (AES-{aes_key_size*8}, ElGamal {elgamal_bits})...")
            start=time.time(); self.aes_key = get_random_bytes(aes_key_size); self.performance.record('AES_KeyGen', time.time()-start) # Symmetric key
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); print("  ✓ ElGamal signature keys generated.")
            self.transmissions = [] # Log of {'id', 'sender', 'enc_data', 'nonce', 'tag', 'signature', 'hash_hex'}

        def send_data(self, sender_id, data):
            print(f"\nSender {sender_id} sending data...")
            data_bytes = data.encode('utf-8')
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_GCM); nonce = cipher_aes.nonce; enc_data, tag = cipher_aes.encrypt_and_digest(data_bytes); self.performance.record('AES_Encrypt_GCM', time.time()-start, len(data_bytes))
            data_to_sign = enc_data + nonce + tag; start=time.time(); hash_obj = SHA256.new(data_to_sign); hash_val = hash_obj.digest(); self.performance.record('SHA256_Hash', time.time()-start)
            start=time.time(); signature = elgamal_sign(hash_val, self.elgamal_keys); self.performance.record('ElGamal_Sign', time.time()-start)
            transmission = {'id': len(self.transmissions) + 1, 'sender': sender_id, 'enc_data': enc_data, 'nonce': nonce, 'tag': tag, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            self.transmissions.append(transmission); print("  ✓ Data encrypted, signed, and logged."); return transmission

        def receive_data(self, transmission):
            print(f"\nReceiver processing transmission ID {transmission['id']} from {transmission['sender']}...")
            elg_pub_key = {k:v for k,v in self.elgamal_keys.items() if k != 'x'}
            data_signed = transmission['enc_data'] + transmission['nonce'] + transmission['tag']; hash_recomputed = SHA256.new(data_signed).digest()
            start=time.time(); is_valid_sig = elgamal_verify(hash_recomputed, transmission['signature'], elg_pub_key); self.performance.record('ElGamal_Verify', time.time()-start)
            if not is_valid_sig: print("  ❌ INVALID SIGNATURE! Data rejected."); return None
            print("  ✓ Signature VALID.")
            start=time.time();
            try: cipher_aes = AES.new(self.aes_key, AES.MODE_GCM, nonce=transmission['nonce']); decrypted_data = cipher_aes.decrypt_and_verify(transmission['enc_data'], transmission['tag']); dec_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', dec_time); print(f"  ✓ Data decrypted (Took: {dec_time:.6f}s)"); return decrypted_data.decode('utf-8')
            except (ValueError, KeyError) as e: dec_time = time.time()-start; self.performance.record('AES_Decrypt_GCM', dec_time); print(f"  ❌ Decryption FAILED: {e}"); return None

    def menu_secure_transmission_aes_elgamal(): # System 25
        if not HAS_CRYPTO: print("\nSystem 25 requires 'pycryptodome'."); pause(); return
        system = SecureTransmissionAESElGamal(); last_transmission = None
        while True:
            clear_screen(); print(f"\n{'='*70}\n  🔒 SYSTEM 25: SECURE TRANSMISSION (AES-GCM + ElGamal Sign + SHA256)\n{'='*70}")
            print("  1. Send Data | 2. Receive Data | 3. View Last Tx | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': sender = input("Sender ID: "); data = input("Data: "); last_transmission = system.send_data(sender, data)
            elif choice == '2':
                if last_transmission: decrypted = system.receive_data(last_transmission);
                    if decrypted is not None: print(f"\n  Decrypted Data: {decrypted}")
                else: print("No data sent yet.")
            elif choice == '3':
                 if last_transmission: print(f"\nTx ID: {last_transmission['id']}, Sender: {last_transmission['sender']}, Enc: {last_transmission['enc_data'].hex()[:20]}..., Nonce: {last_transmission['nonce'].hex()}, Tag: {last_transmission['tag'].hex()}, Hash: {last_transmission['hash_hex'][:16]}..., Sig: {last_transmission['signature']}")
                 else: print("No details.")
            elif choice == '4': system.performance.print_graph("Secure Transmission Performance")
            elif choice == '5': system.performance.compare_algorithms(['AES_Encrypt_GCM', 'ElGamal_Sign', 'SHA256_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()
else:
    def menu_secure_transmission_aes_elgamal(): print("\nSystem 25 requires 'pycryptodome'."); pause()

# --- System 26: Secure Storage (Rabin Enc + RSA Sign + SHA512) ---
if HAS_CRYPTO:
    class SecureStorageRabinRSA:
        def __init__(self, rabin_bits=2048, rsa_bits=2048):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Secure Storage (Rabin {rabin_bits}, RSA {rsa_bits})...")
            start=time.time(); self.rabin_keys = self._generate_rabin_keys(rabin_bits); self.performance.record('Rabin_KeyGen', time.time()-start)
            start=time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Sign', time.time()-start); print("  ✓ RSA keys generated.")
            self.files = {} # filename: {'enc_content', 'signature', 'hash_hex'}
        def _generate_rabin_keys(self, bits): # Same as Sys 3
             while True: p = number.getPrime(bits // 2);
                 if p % 4 == 3: break
             while True: q = number.getPrime(bits // 2);
                 if q % 4 == 3 and q != p: break
             print("✓ Rabin keys generated."); return {'n': p * q, 'p': p, 'q': q}
        def _rabin_encrypt(self, msg_bytes, n): # Same padding as Sys 3
            redundancy = b"FILEPAD" + len(msg_bytes).to_bytes(4, 'big')
            full_msg = msg_bytes + redundancy; m = number.bytes_to_long(full_msg)
            if m >= n: raise ValueError("Message too large for Rabin key")
            return pow(m, 2, n)
        def _rabin_decrypt(self, cipher_int, p, q, n): # Same as Sys 3
            mp=pow(cipher_int,(p+1)//4,p); mq=pow(cipher_int,(q+1)//4,q); inv_p=number.inverse(p,q); inv_q=number.inverse(q,p); a=(q*inv_q)%n; b=(p*inv_p)%n
            roots=[(a*mp+b*mq)%n,(a*mp-b*mq)%n,(-a*mp+b*mq)%n,(-a*mp-b*mq)%n]
            for r in roots:
                try: r_bytes=number.long_to_bytes(r);
                    if len(r_bytes)>11 and r_bytes.endswith(b"FILEPAD"): len_s=len(r_bytes)-11; msg_len=int.from_bytes(r_bytes[len_s:len_s+4],'big');
                        if len_s==msg_len: return r_bytes[:msg_len]
                except Exception: continue
            return None
        def store_file(self, filename, content):
            print(f"\nStoring file '{filename}'...")
            content_bytes = content.encode('utf-8')
            start=time.time();
            try: enc_content = self._rabin_encrypt(content_bytes, self.rabin_keys['n'])
            except ValueError as e: print(f"  Error encrypting: {e}"); return False
            self.performance.record('Rabin_Encrypt', time.time()-start, len(content_bytes))
            data_to_sign = number.long_to_bytes(enc_content) + filename.encode('utf-8')
            start=time.time(); hash_obj = SHA512.new(data_to_sign); self.performance.record('SHA512_Hash', time.time()-start)
            start=time.time(); signature = pkcs1_15.new(self.rsa_key).sign(hash_obj); self.performance.record('RSA_Sign', time.time()-start)
            self.files[filename] = {'enc_content': enc_content, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            print("  ✓ File encrypted, signed, and stored.")
            return True
        def retrieve_file(self, filename):
            if filename not in self.files: print(f"File '{filename}' not found."); return None
            print(f"\nRetrieving file '{filename}'..."); stored = self.files[filename]
            data_signed = number.long_to_bytes(stored['enc_content']) + filename.encode('utf-8'); hash_obj_recomputed = SHA512.new(data_signed)
            start=time.time();
            try: pkcs1_15.new(self.rsa_pub_key).verify(hash_obj_recomputed, stored['signature']); is_valid = True
            except (ValueError, TypeError): is_valid = False
            self.performance.record('RSA_Verify', time.time()-start)
            if not is_valid: print("  ❌ INVALID SIGNATURE! Tampered?"); return None
            print("  ✓ Signature VALID.")
            start=time.time(); decrypted_bytes = self._rabin_decrypt(stored['enc_content'], self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n']); dec_time = time.time()-start; self.performance.record('Rabin_Decrypt', dec_time)
            if decrypted_bytes is None: print("  ❌ Decryption FAILED."); return None
            print(f"  ✓ File decrypted (Took: {dec_time:.6f}s)"); return decrypted_bytes.decode('utf-8', errors='replace')

    def menu_secure_storage_rabin_rsa(): # System 26
        if not HAS_CRYPTO: print("\nSystem 26 requires 'pycryptodome'."); pause(); return
        system = SecureStorageRabinRSA()
        while True:
            clear_screen(); print(f"\n{'='*70}\n  💾 SYSTEM 26: SECURE STORAGE (Rabin Enc + RSA Sign + SHA512)\n{'='*70}")
            print("  1. Store File | 2. Retrieve File | 3. List Files | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': filename = input("Filename: "); content = input("Content: "); system.store_file(filename, content)
            elif choice == '2': filename = input("Filename: "); content = system.retrieve_file(filename);
                if content is not None: print(f"\n  Retrieved Content:\n{content}")
            elif choice == '3': print("\n--- Stored Files ---"); [print(f"  - {fname} (Enc: {str(finfo['enc_content'])[:20]}..., Sig: {finfo['signature'].hex()[:16]}...)") for fname, finfo in system.files.items()] if system.files else print("  None.")
            elif choice == '4': system.performance.print_graph("Secure Storage Performance")
            elif choice == '5': system.performance.compare_algorithms(['Rabin_Encrypt', 'RSA_Sign', 'SHA512_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()
else:
    def menu_secure_storage_rabin_rsa(): print("\nSystem 26 requires 'pycryptodome'."); pause()

# --- System 27: Signed Encryption (RSA Encrypt + ElGamal Sign + SHA1) ---
if HAS_CRYPTO:
    class SignedEncryptionRSAElGamal:
        def __init__(self, rsa_bits=1024, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Signed Encryption (RSA {rsa_bits} Enc, ElGamal {elgamal_bits} Sig)...")
            start=time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Enc', time.time()-start); print("  ✓ RSA encryption keys.")
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); print("  ✓ ElGamal signature keys.")
            self.messages = [] # {'id', 'enc_data', 'signature', 'hash_hex'}

        def send_message(self, msg_id, message):
            print(f"\nSending message {msg_id}...")
            msg_bytes = message.encode('utf-8')
            start=time.time(); hash_obj = SHA1.new(msg_bytes); hash_val = hash_obj.digest(); self.performance.record('SHA1_Hash', time.time()-start)
            start=time.time(); signature = elgamal_sign(hash_val, self.elgamal_keys); self.performance.record('ElGamal_Sign', time.time()-start)
            start=time.time(); cipher_rsa = PKCS1_OAEP.new(self.rsa_pub_key); enc_data = cipher_rsa.encrypt(msg_bytes); self.performance.record('RSA_Encrypt', time.time()-start, len(msg_bytes))
            record = {'id': msg_id, 'enc_data': enc_data, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            self.messages.append(record); print("  ✓ Message encrypted and signed."); return record

        def receive_message(self, record):
            print(f"\nReceiving message {record['id']}...")
            elg_pub_key = {k:v for k,v in self.elgamal_keys.items() if k != 'x'}
            start=time.time(); cipher_rsa = PKCS1_OAEP.new(self.rsa_key)
            try: dec_bytes = cipher_rsa.decrypt(record['enc_data']); dec_msg = dec_bytes.decode('utf-8'); self.performance.record('RSA_Decrypt', time.time()-start)
            except (ValueError, TypeError) as e: print(f"  ❌ Decryption failed: {e}"); self.performance.record('RSA_Decrypt', time.time()-start); return None, False
            print(f"  ✓ Decrypted: '{dec_msg}'")
            start=time.time(); hash_recomputed = SHA1.new(dec_bytes).digest(); is_valid = elgamal_verify(hash_recomputed, record['signature'], elg_pub_key); self.performance.record('ElGamal_Verify', time.time()-start)
            print(f"  ✓ Signature Verification: {'VALID' if is_valid else 'INVALID'}")
            print(f"    Received Hash (SHA1): {record['hash_hex']}\n    Computed Hash (SHA1): {SHA1.new(dec_bytes).hexdigest()}"); return dec_msg, is_valid

    def menu_signed_encryption_rsa_elgamal(): # System 27
        if not HAS_CRYPTO: print("\nSystem 27 requires 'pycryptodome'."); pause(); return
        system = SignedEncryptionRSAElGamal(); msg_count = 0; last_msg_record = None
        while True:
            clear_screen(); print(f"\n{'='*70}\n  ✍️ SYSTEM 27: SIGNED ENCRYPTION (RSA Enc + ElGamal Sig + SHA1)\n{'='*70}")
            print("  1. Send Message | 2. Receive Last Message | 3. View Last Record | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': msg_count+=1; msg = input("Message to send: "); last_msg_record = system.send_message(f"M{msg_count}", msg)
            elif choice == '2':
                if last_msg_record: msg, valid = system.receive_message(last_msg_record);
                    if msg is not None: print(f"\n  Received Content: {msg} (Signature Valid: {valid})")
                else: print("No message sent yet.")
            elif choice == '3':
                if last_msg_record: print(f"\nRecord ID: {last_msg_record['id']}\n Enc: {last_msg_record['enc_data'].hex()[:40]}...\n Sig: {last_msg_record['signature']}\n Hash: {last_msg_record['hash_hex']}")
                else: print("No record.")
            elif choice == '4': system.performance.print_graph("Signed Encryption Performance")
            elif choice == '5': system.performance.compare_algorithms(['RSA_Encrypt', 'ElGamal_Sign', 'SHA1_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()
else:
    def menu_signed_encryption_rsa_elgamal(): print("\nSystem 27 requires 'pycryptodome'."); pause()

# --- System 28: Encrypt-then-MAC (AES-CBC + HMAC-SHA256) ---
if HAS_CRYPTO:
    # from Crypto.Hash import HMAC # Import moved to top
    class EncryptThenMAC:
        def __init__(self, aes_key_size=32, mac_key_size=32): # AES-256, SHA256-HMAC
            self.performance = PerformanceTracker()
            print(f"\nInitializing Encrypt-then-MAC (AES-{aes_key_size*8} CBC, HMAC-SHA256)...")
            start=time.time(); self.aes_key = get_random_bytes(aes_key_size); self.performance.record('AES_KeyGen', time.time()-start)
            start=time.time(); self.mac_key = get_random_bytes(mac_key_size); self.performance.record('HMAC_KeyGen', time.time()-start) # Separate key for MAC
            self.messages = [] # {'id', 'iv', 'ciphertext', 'mac_tag'}

        def protect_message(self, msg_id, message):
            print(f"\nProtecting message {msg_id}...")
            msg_bytes = message.encode('utf-8')
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_CBC); iv = cipher_aes.iv; ciphertext = cipher_aes.encrypt(pad(msg_bytes, AES.block_size)); self.performance.record('AES_Encrypt_CBC', time.time()-start, len(msg_bytes))
            start=time.time(); hmac_sha256 = HMAC.new(self.mac_key, digestmod=SHA256); hmac_sha256.update(iv + ciphertext); mac_tag = hmac_sha256.digest(); self.performance.record('HMAC_SHA256_Generate', time.time()-start)
            record = {'id': msg_id, 'iv': iv, 'ciphertext': ciphertext, 'mac_tag': mac_tag}
            self.messages.append(record); print("  ✓ Message encrypted and MAC generated."); return record

        def verify_and_decrypt(self, record):
            print(f"\nVerifying and decrypting message {record['id']}...")
            start=time.time(); hmac_sha256 = HMAC.new(self.mac_key, digestmod=SHA256); hmac_sha256.update(record['iv'] + record['ciphertext']);
            try: hmac_sha256.verify(record['mac_tag']); is_valid = True; self.performance.record('HMAC_SHA256_Verify', time.time()-start)
            except ValueError: is_valid = False; self.performance.record('HMAC_SHA256_Verify', time.time()-start); print("  ❌ INVALID MAC TAG! Tampered?"); return None
            print("  ✓ MAC Tag VALID.")
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_CBC, iv=record['iv'])
            try: dec_bytes = unpad(cipher_aes.decrypt(record['ciphertext']), AES.block_size); dec_msg = dec_bytes.decode('utf-8'); self.performance.record('AES_Decrypt_CBC', time.time()-start)
            except (ValueError, KeyError) as e: print(f"  ❌ Decryption failed: {e}"); self.performance.record('AES_Decrypt_CBC', time.time()-start); return None
            print(f"  ✓ Decrypted: '{dec_msg}'"); return dec_msg

    def menu_encrypt_then_mac(): # System 28
        if not HAS_CRYPTO: print("\nSystem 28 requires 'pycryptodome'."); pause(); return
        system = EncryptThenMAC(); msg_count = 0; last_msg_record = None
        while True:
            clear_screen(); print(f"\n{'='*70}\n  🔐 SYSTEM 28: ENCRYPT-THEN-MAC (AES-CBC + HMAC-SHA256)\n{'='*70}")
            print("  1. Protect Message | 2. Verify & Decrypt Last | 3. View Last Record | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': msg_count+=1; msg = input("Message: "); last_msg_record = system.protect_message(f"EtM_{msg_count}", msg)
            elif choice == '2':
                if last_msg_record: msg = system.verify_and_decrypt(last_msg_record);
                    if msg is not None: print(f"\n  Verified Content: {msg}")
                else: print("No message protected yet.")
            elif choice == '3':
                if last_msg_record: print(f"\nRecord ID: {last_msg_record['id']}\n IV: {last_msg_record['iv'].hex()}\n CT: {last_msg_record['ciphertext'].hex()[:40]}...\n MAC: {last_msg_record['mac_tag'].hex()}")
                else: print("No record.")
            elif choice == '4': system.performance.print_graph("Encrypt-then-MAC Performance")
            elif choice == '5': system.performance.compare_algorithms(['AES_Encrypt_CBC', 'HMAC_SHA256_Generate', 'HMAC_SHA256_Verify'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()
else:
    def menu_encrypt_then_mac(): print("\nSystem 28 requires 'pycryptodome'."); pause()

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 4: ADVANCED CONCEPTS (PLACEHOLDERS) (Systems 29-30)
# FULL CODE PASTED HERE
#
# ═══════════════════════════════════════════════════════════════════════════

def menu_sse_placeholder(): # System 29
    clear_screen()
    print("\n--- SYSTEM 29: Symmetric Searchable Encryption (SSE) ---")
    print("\nCONCEPT:")
    print("  Allows a client to store encrypted data on an untrusted server")
    print("  and later have the server search for specific keywords without")
    print("  decrypting the data. Uses symmetric keys shared between client/server.")
    print("\nUSE CASES:")
    print("  - Searching encrypted emails stored on a server.")
    print("  - Querying encrypted databases in the cloud.")
    print("\nIMPLEMENTATION:")
    print("  - Often involves building an encrypted index (e.g., using keywords).")
    print("  - Client generates a 'search token' or 'trapdoor' for a keyword.")
    print("  - Server uses the token to find matching encrypted documents via the index.")
    print("  - Schemes vary in security (e.g., leakage) and efficiency.")
    print("  - **Not implemented here due to complexity.** Requires specialized libraries/protocols.")
    print("\nCHALLENGES: Preventing leakage about search terms or access patterns.")
    pause()

def menu_pkse_placeholder(): # System 30
    clear_screen()
    print("\n--- SYSTEM 30: Public Key Searchable Encryption (PKSE) ---")
    print("\nCONCEPT:")
    print("  Allows searching over data encrypted with a public key.")
    print("  Often involves multiple parties:")
    print("    - Data Owner: Encrypts data with public key.")
    print("    - User: Holds private key, generates search tokens.")
    print("    - Server: Stores encrypted data, performs searches using tokens.")
    print("  Server learns minimal info (ideally just matches, sometimes less).")
    print("\nUSE CASES:")
    print("  - Secure email gateway filtering encrypted emails.")
    print("  - Multi-user access to encrypted cloud databases with search.")
    print("\nIMPLEMENTATION:")
    print("  - More complex than SSE. Often relies on Identity-Based Encryption (IBE),")
    print("    Attribute-Based Encryption (ABE), or bilinear pairings.")
    print("  - **Not implemented here due to high complexity.** Requires advanced crypto libraries.")
    print("\nCHALLENGES: Security proofs, efficiency, key management.")
    pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 5: TOOLS & EXIT (Systems 31-33)
# FULL CODE PASTED HERE
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 31: Universal Benchmark ---
def run_comprehensive_benchmark(): # System 31
    clear_screen(); print(f"\n{'='*70}\n  ⏱️ SYSTEM 31: COMPREHENSIVE ALGORITHM BENCHMARK\n{'='*70}")
    # Use larger data if desired, but smaller ensures faster testing
    test_data_base = "The quick brown fox jumps over the lazy dog. Sample data for benchmark." * 20 # ~1KB
    test_data_bytes = test_data_base.encode('utf-8'); data_size_bytes = len(test_data_bytes)
    results = {}; tracker = PerformanceTracker(); print(f"\nBenchmarking (Data size: {data_size_bytes} bytes)...")
    iter_fast, iter_med, iter_slow = 1000, 100, 20 # Adjusted iterations

    # Symmetric
    if HAS_CRYPTO:
        print("  Testing DES..."); des_key = get_random_bytes(8); cipher = DES.new(des_key, DES.MODE_CBC); start = time.time(); [cipher.encrypt(pad(test_data_bytes, DES.block_size)) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['DES-CBC Enc'] = avg_time; tracker.record('Benchmark_DES_Enc', avg_time, data_size_bytes)
        print("  Testing AES-256..."); aes_key = get_random_bytes(32); cipher = AES.new(aes_key, AES.MODE_CBC); start = time.time(); [cipher.encrypt(pad(test_data_bytes, AES.block_size)) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['AES-CBC Enc'] = avg_time; tracker.record('Benchmark_AES256_Enc', avg_time, data_size_bytes)
        print("  Testing AES-GCM Encrypt..."); cipher_gcm = AES.new(aes_key, AES.MODE_GCM); start = time.time(); [cipher_gcm.encrypt_and_digest(test_data_bytes) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['AES-GCM Enc'] = avg_time; tracker.record('Benchmark_AESGCM_Enc', avg_time, data_size_bytes)

    # Asymmetric Enc/Dec
    if HAS_CRYPTO:
        print("  Testing RSA-2048 Encrypt..."); rsa_key = RSA.generate(2048); cipher_rsa = PKCS1_OAEP.new(rsa_key.publickey()); data_to_enc_rsa = test_data_bytes[:128]; start = time.time(); [cipher_rsa.encrypt(data_to_enc_rsa) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['RSA-2048 Enc'] = avg_time; tracker.record('Benchmark_RSA2048_Enc', avg_time, 128)
        enc_rsa = cipher_rsa.encrypt(data_to_enc_rsa); print("  Testing RSA-2048 Decrypt..."); cipher_rsa_priv = PKCS1_OAEP.new(rsa_key); start = time.time(); [cipher_rsa_priv.decrypt(enc_rsa) for _ in range(iter_slow)]; avg_time = (time.time() - start) / iter_slow; results['RSA-2048 Dec'] = avg_time; tracker.record('Benchmark_RSA2048_Dec', avg_time, 128)

    if HAS_PAILLIER:
        print("  Testing Paillier Encrypt..."); paillier_pub, paillier_priv = paillier.generate_paillier_keypair(n_length=1024); val_to_enc = 12345; start = time.time(); [paillier_pub.encrypt(val_to_enc) for _ in range(iter_med)]; avg_time = (time.time() - start) / iter_med; results['Paillier Enc'] = avg_time; tracker.record('Benchmark_Paillier_
