#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
    ICT3141 ULTIMATE EXAM SCRIPT - ALL SCENARIOS
    
    Part 1: 12 Complete 3-Algorithm Systems (Systems 1-12)
    Part 2: 6 Classical & Lab Scenarios (Systems 13-18)
    
    Total Systems: 18 Complete Scenarios
    All with Menu-Driven Interfaces and Performance Graphs
═══════════════════════════════════════════════════════════════════════════
"""

import sys
import random
import time
import hashlib
import base64
import json
from datetime import datetime
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
# LIBRARY CHECKS & IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  ICT3141 ULTIMATE EXAM SCRIPT - INITIALIZING")
print("="*70)

try:
    from Crypto.Cipher import DES, DES3, AES, PKCS1_OAEP
    from Crypto.PublicKey import RSA
    from Crypto.Hash import SHA256, SHA512, SHA1, MD5
    from Crypto.Util import number
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Random import get_random_bytes
    from Crypto.Signature import pkcs1_15
    HAS_CRYPTO = True
    print("  ✓ PyCryptodome loaded (DES, 3DES, AES, RSA, Hashes, Signatures)")
except ImportError:
    HAS_CRYPTO = False
    print("  ✗ PyCryptodome not installed!")
    print("\n  Install with: pip install pycryptodome")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
    print("  ✓ NumPy loaded (Hill Cipher systems enabled)")
except ImportError:
    HAS_NUMPY = False
    print("  ⚠ NumPy not available (Hill Cipher systems disabled)")

try:
    from phe import paillier
    HAS_PAILLIER = True
    print("  ✓ Paillier (phe) loaded (E-Voting system enabled)")
except ImportError:
    HAS_PAILLIER = False
    print("  ⚠ Paillier not available (E-Voting system disabled)")

print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# (Dependencies for most systems)
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceTracker:
    """Track and visualize performance metrics"""
    
    def __init__(self):
        self.metrics = []
    
    def record(self, operation, time_taken, data_size=0):
        """Record a performance metric"""
        self.metrics.append({
            'operation': operation,
            'time': time_taken,
            'size': data_size,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_stats(self, operation=None):
        """Get statistics for operations"""
        if operation:
            data = [m for m in self.metrics if m['operation'] == operation]
        else:
            data = self.metrics
        
        if not data:
            return None
        
        times = [m['time'] for m in data]
        return {
            'count': len(times),
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }
    
    def print_graph(self, title="Performance Analysis"):
        """Print ASCII bar graph"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print("="*70)
        
        # Group by operation
        ops = defaultdict(list)
        for m in self.metrics:
            ops[m['operation']].append(m['time'])
        
        if not ops:
            print("  No data recorded yet")
            return
        
        print("\nOperation Statistics:")
        for op, times in sorted(ops.items()):
            avg = sum(times) / len(times)
            print(f"\n  {op}:")
            print(f"    Count:   {len(times)}")
            print(f"    Average: {avg:.6f}s")
            print(f"    Min:     {min(times):.6f}s")
            print(f"    Max:     {max(times):.6f}s")
            # Scale graph bar
            scale = 2000
            bar = '█' * int(avg * scale)
            if int(avg * scale) == 0 and avg > 0:
                bar = '·'
            print(f"    Graph:   {bar}")
        
        print("\n" + "="*70)
    
    def compare_algorithms(self, alg1, alg2, alg3):
        """Compare three algorithms"""
        stats = {}
        for alg in [alg1, alg2, alg3]:
            s = self.get_stats(alg)
            if s:
                stats[alg] = s['average']
        
        if len(stats) < 2:
            print("Not enough data for comparison")
            return
        
        print(f"\n{'='*70}")
        print("  ALGORITHM COMPARISON")
        print("="*70)
        
        sorted_algs = sorted(stats.items(), key=lambda x: x[1])
        
        print("\nSpeed Ranking (fastest to slowest):")
        scale = 5000
        for i, (alg, time) in enumerate(sorted_algs, 1):
            bar = '█' * int(time * scale)
            if int(time * scale) == 0 and time > 0:
                bar = '·'
            print(f"  {i}. {alg:20s}: {time:.6f}s")
            print(f"     {bar}")
        
        fastest = sorted_algs[0][1]
        if fastest == 0:
            print("\nFastest algorithm is too fast to measure relative performance.")
            print("="*70)
            return

        print("\nRelative Performance:")
        for alg, time in sorted_algs[1:]:
            ratio = time / fastest
            print(f"  {alg} is {ratio:.1f}x slower than {sorted_algs[0][0]}")
        
        print("="*70)

def gcd(a, b):
    """Greatest Common Divisor"""
    while b:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    """Modular inverse"""
    a = a % m
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None

def matrix_mod_inv(matrix, modulus):
    """Find the modular inverse of a matrix (requires NumPy)"""
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for matrix operations.")
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = mod_inverse(det % modulus, modulus)
    if det_inv is None:
        raise ValueError(f"Matrix determinant {det} is not invertible mod {modulus} (gcd={gcd(det % modulus, modulus)})")
    
    adjugate = np.linalg.inv(matrix) * det
    inv = (det_inv * np.round(adjugate)) % modulus
    return inv.astype(int)


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 1: COMPLETE 3-ALGORITHM SYSTEMS
#
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 1: DES + RSA + SHA-256 (Secure Email)
# ═══════════════════════════════════════════════════════════════════════════

class SecureEmailSystem:
    """Complete email system with DES, RSA, and SHA-256"""
    
    def __init__(self):
        self.users = {}
        self.mailboxes = {}
        self.performance = PerformanceTracker()
    
    def register_user(self, user_id):
        """Generate RSA keys for user"""
        start = time.time()
        key = RSA.generate(2048)
        self.users[user_id] = {
            'private_key': key,
            'public_key': key.publickey()
        }
        self.mailboxes[user_id] = []
        self.performance.record('RSA_KeyGen', time.time() - start)
        print(f"✓ {user_id} registered with RSA-2048 keys")
    
    def send_email(self, sender, recipient, subject, body):
        """Encrypt email: DES for content, RSA for key, SHA-256 for integrity"""
        if recipient not in self.users:
            print("Recipient not found!")
            return None
        
        # DES encryption
        start = time.time()
        des_key = base64.b64encode(get_random_bytes(8)).decode()[:8]
        cipher_des = DES.new(des_key.encode(), DES.MODE_ECB)
        content = f"{subject}||{body}"
        encrypted_body = cipher_des.encrypt(pad(content.encode(), DES.block_size))
        des_time = time.time() - start
        self.performance.record('DES_Encrypt', des_time, len(content))
        
        # SHA-256 hash
        start = time.time()
        email_hash = SHA256.new(content.encode()).hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA256_Hash', sha_time)
        
        # RSA key encryption
        start = time.time()
        cipher_rsa = PKCS1_OAEP.new(self.users[recipient]['public_key'])
        encrypted_key = cipher_rsa.encrypt(des_key.encode())
        rsa_time = time.time() - start
        self.performance.record('RSA_Encrypt', rsa_time)
        
        email = {
            'id': len(self.mailboxes[recipient]) + 1,
            'from': sender,
            'encrypted_key': base64.b64encode(encrypted_key).decode(),
            'encrypted_body': base64.b64encode(encrypted_body).decode(),
            'hash': email_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        self.mailboxes[recipient].append(email)
        
        print(f"\n✓ Email sent!")
        print(f"  DES encryption: {des_time:.6f}s")
        print(f"  RSA key encrypt: {rsa_time:.6f}s")
        print(f"  SHA-256 hash: {sha_time:.6f}s")
        print(f"  Total: {des_time + rsa_time + sha_time:.6f}s")
        
        return email['id']
    
    def read_email(self, user_id, email_id):
        """Decrypt email and verify integrity"""
        email = next((e for e in self.mailboxes.get(user_id, []) if e['id'] == email_id), None)
        if not email:
            print("Email not found!")
            return None
        
        # RSA key decryption
        start = time.time()
        cipher_rsa = PKCS1_OAEP.new(self.users[user_id]['private_key'])
        des_key = cipher_rsa.decrypt(base64.b64decode(email['encrypted_key'])).decode()
        rsa_time = time.time() - start
        self.performance.record('RSA_Decrypt', rsa_time)
        
        # DES decryption
        start = time.time()
        cipher_des = DES.new(des_key.encode(), DES.MODE_ECB)
        decrypted = unpad(cipher_des.decrypt(base64.b64decode(email['encrypted_body'])), DES.block_size).decode()
        des_time = time.time() - start
        self.performance.record('DES_Decrypt', des_time)
        
        # SHA-256 verification
        start = time.time()
        computed_hash = SHA256.new(decrypted.encode()).hexdigest()
        verified = computed_hash == email['hash']
        sha_time = time.time() - start
        self.performance.record('SHA256_Verify', sha_time)
        
        subject, body = decrypted.split('||', 1)
        
        print(f"\n✓ Email decrypted!")
        print(f"  RSA key decrypt: {rsa_time:.6f}s")
        print(f"  DES decryption: {des_time:.6f}s")
        print(f"  SHA-256 verify: {sha_time:.6f}s")
        print(f"  Hash verified: {verified}")
        print(f"\n  Subject: {subject}")
        print(f"  Body: {body}")
        
        return decrypted

def menu_email_system():
    """Menu for secure email system"""
    system = SecureEmailSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 1: SECURE EMAIL (DES + RSA + SHA-256)")
        print("="*70)
        print("1. Register User")
        print("2. Send Email")
        print("3. Read Email")
        print("4. View Mailbox")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            user_id = input("User ID: ")
            system.register_user(user_id)
        elif choice == '2':
            sender = input("From: ")
            recipient = input("To: ")
            subject = input("Subject: ")
            body = input("Body: ")
            system.send_email(sender, recipient, subject, body)
        elif choice == '3':
            user = input("Your ID: ")
            email_id = int(input("Email ID: "))
            system.read_email(user, email_id)
        elif choice == '4':
            user = input("Your ID: ")
            if user in system.mailboxes:
                for email in system.mailboxes[user]:
                    print(f"\n[{email['id']}] From: {email['from']}")
        elif choice == '5':
            system.performance.print_graph("Email System Performance")
        elif choice == '6':
            system.performance.compare_algorithms('DES_Encrypt', 'RSA_Encrypt', 'SHA256_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 2: AES + ElGamal + SHA-512 (Banking)
# ═══════════════════════════════════════════════════════════════════════════

class BankingSystem:
    """Financial transactions with AES, ElGamal, SHA-512"""
    
    def __init__(self):
        self.aes_key = hashlib.sha256(b"BankMasterKey").digest()
        self.elgamal_keys = {}
        self.accounts = {}
        self.transactions = []
        self.performance = PerformanceTracker()
    
    def create_account(self, customer_id, balance):
        """Create account with ElGamal signing keys"""
        start = time.time()
        
        bits = 512
        while True:
            q = number.getPrime(bits - 1)
            p = 2 * q + 1
            if number.isPrime(p):
                break
        
        while True:
            g = random.randint(2, p - 2)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                break
        
        x = random.randint(2, p - 2)
        y = pow(g, x, p)
        
        self.elgamal_keys[customer_id] = {'p': p, 'g': g, 'x': x, 'y': y}
        self.accounts[customer_id] = {'balance': balance}
        
        keygen_time = time.time() - start
        self.performance.record('ElGamal_KeyGen', keygen_time)
        
        print(f"✓ Account created for {customer_id}")
        print(f"  Balance: ${balance}")
        print(f"  Key generation: {keygen_time:.6f}s")
    
    def create_transaction(self, from_cust, to_cust, amount, description):
        """Create encrypted and signed transaction"""
        if from_cust not in self.accounts:
            print("Account not found!")
            return None
        
        txn_data = f"{from_cust}|{to_cust}|{amount}|{description}"
        
        # SHA-512 hash
        start = time.time()
        h = SHA512.new(txn_data.encode())
        hash_hex = h.hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA512_Hash', sha_time)
        
        # ElGamal signature
        start = time.time()
        keys = self.elgamal_keys[from_cust]
        p, g, x = keys['p'], keys['g'], keys['x']
        h_int = int.from_bytes(h.digest(), 'big') % (p - 1)
        
        while True:
            k = random.randint(2, p - 2)
            if gcd(k, p - 1) == 1:
                break
        
        r = pow(g, k, p)
        k_inv = number.inverse(k, p - 1)
        s = (k_inv * (h_int - x * r)) % (p - 1)
        
        elg_time = time.time() - start
        self.performance.record('ElGamal_Sign', elg_time)
        
        # AES encryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(pad(txn_data.encode(), AES.block_size))
        aes_time = time.time() - start
        self.performance.record('AES_Encrypt', aes_time, len(txn_data))
        
        txn = {
            'id': len(self.transactions) + 1,
            'from': from_cust,
            'to': to_cust,
            'amount': amount,
            'encrypted': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'tag': base64.b64encode(tag).decode(),
            'hash': hash_hex,
            'sig_r': r,
            'sig_s': s,
            'status': 'pending'
        }
        
        self.transactions.append(txn)
        
        print(f"\n✓ Transaction created!")
        print(f"  AES encryption: {aes_time:.6f}s")
        print(f"  ElGamal signing: {elg_time:.6f}s")
        print(f"  SHA-512 hash: {sha_time:.6f}s")
        print(f"  Total: {aes_time + elg_time + sha_time:.6f}s")
        
        return txn['id']
    
    def process_transaction(self, txn_id):
        """Process and verify transaction"""
        txn = next((t for t in self.transactions if t['id'] == txn_id), None)
        if not txn:
            print("Transaction not found!")
            return False
        
        # AES decryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX, nonce=base64.b64decode(txn['nonce']))
        decrypted = unpad(cipher.decrypt(base64.b64decode(txn['encrypted'])), AES.block_size).decode()
        cipher.verify(base64.b64decode(txn['tag']))
        aes_time = time.time() - start
        self.performance.record('AES_Decrypt', aes_time)
        
        # ElGamal verification
        start = time.time()
        keys = self.elgamal_keys[txn['from']]
        p, g, y = keys['p'], keys['g'], keys['y']
        
        h = SHA512.new(decrypted.encode())
        h_int = int.from_bytes(h.digest(), 'big') % (p - 1)
        
        v1 = (pow(y, txn['sig_r'], p) * pow(txn['sig_r'], txn['sig_s'], p)) % p
        v2 = pow(g, h_int, p)
        verified = v1 == v2
        
        elg_time = time.time() - start
        self.performance.record('ElGamal_Verify', elg_time)
        
        if verified:
            self.accounts[txn['from']]['balance'] -= txn['amount']
            if txn['to'] not in self.accounts:
                self.accounts[txn['to']] = {'balance': 0}
            self.accounts[txn['to']]['balance'] += txn['amount']
            txn['status'] = 'completed'
            
            print(f"\n✓ Transaction processed!")
            print(f"  AES decryption: {aes_time:.6f}s")
            print(f"  ElGamal verify: {elg_time:.6f}s")
            print(f"  Signature: VALID")
            return True
        else:
            print(f"\n✗ Transaction rejected - Invalid signature")
            return False

def menu_banking_system():
    """Menu for banking system"""
    system = BankingSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 2: BANKING (AES-256 + ElGamal + SHA-512)")
        print("="*70)
        print("1. Create Account")
        print("2. View Balance")
        print("3. Create Transaction")
        print("4. Process Transaction")
        print("5. View All Transactions")
        print("6. Performance Analysis")
        print("7. Algorithm Comparison")
        print("8. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            cust = input("Customer ID: ")
            bal = float(input("Initial balance: $"))
            system.create_account(cust, bal)
        elif choice == '2':
            cust = input("Customer ID: ")
            if cust in system.accounts:
                print(f"\nBalance: ${system.accounts[cust]['balance']}")
        elif choice == '3':
            from_c = input("From: ")
            to_c = input("To: ")
            amt = float(input("Amount: $"))
            desc = input("Description: ")
            system.create_transaction(from_c, to_c, amt, desc)
        elif choice == '4':
            txn_id = int(input("Transaction ID: "))
            system.process_transaction(txn_id)
        elif choice == '5':
            for txn in system.transactions:
                print(f"\n[{txn['id']}] {txn['from']} → {txn['to']}: ${txn['amount']} ({txn['status']})")
        elif choice == '6':
            system.performance.print_graph("Banking System Performance")
        elif choice == '7';:
            system.performance.compare_algorithms('AES_Encrypt', 'ElGamal_Sign', 'SHA512_Hash')
        elif choice == '8':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 3: Rabin + RSA + MD5 (Cloud Storage)
# ═══════════════════════════════════════════════════════════════════════════

class CloudStorageSystem:
    """Secure cloud storage with Rabin, RSA, MD5"""
    
    def __init__(self):
        self.rabin_keys = self._init_rabin()
        self.rsa_keys = {}
        self.files = []
        self.performance = PerformanceTracker()
    
    def _init_rabin(self):
        """Initialize Rabin keys"""
        start = time.time()
        bits = 1024
        while True:
            p = number.getPrime(bits // 2)
            if p % 4 == 3:
                break
        while True:
            q = number.getPrime(bits // 2)
            if q % 4 == 3 and q != p:
                break
        n = p * q
        self.performance.record('Rabin_KeyGen', time.time() - start)
        return {'n': n, 'p': p, 'q': q}
    
    def register_user(self, user_id):
        """Generate RSA keys"""
        start = time.time()
        key = RSA.generate(2048)
        self.rsa_keys[user_id] = {
            'private': key,
            'public': key.publickey()
        }
        self.performance.record('RSA_KeyGen', time.time() - start)
        print(f"✓ {user_id} registered")
    
    def _rabin_encrypt(self, message):
        """Encrypt with Rabin"""
        msg_bytes = message.encode()
        redundancy = b"CLOUD" + len(msg_bytes).to_bytes(2, 'big')
        full_msg = redundancy + msg_bytes
        m = int.from_bytes(full_msg, 'big')
        if m >= self.rabin_keys['n']:
            raise ValueError("Message too long")
        return pow(m, 2, self.rabin_keys['n'])
    
    def _rabin_decrypt(self, c):
        """Decrypt Rabin"""
        p, q, n = self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n']
        mp = pow(c, (p + 1) // 4, p)
        mq = pow(c, (q + 1) // 4, q)
        inv_p = number.inverse(p, q)
        inv_q = number.inverse(q, p)
        a = (q * inv_q) % n
        b = (p * inv_p) % n
        roots = [
            (a * mp + b * mq) % n,
            (a * mp - b * mq) % n,
            (-a * mp + b * mq) % n,
            (-a * mp - b * mq) % n
        ]
        for r in roots:
            try:
                msg_bytes = r.to_bytes((r.bit_length() + 7) // 8, 'big')
                if msg_bytes[:5] == b"CLOUD":
                    length = int.from_bytes(msg_bytes[5:7], 'big')
                    return msg_bytes[7:7+length].decode()
            except:
                continue
        return None
    
    def upload_file(self, owner, filename, content):
        """Upload encrypted file"""
        if owner not in self.rsa_keys:
            print("User not registered!")
            return None
        
        # MD5 hash
        start = time.time()
        file_hash = MD5.new(content.encode()).hexdigest()
        md5_time = time.time() - start
        self.performance.record('MD5_Hash', md5_time)
        
        # Rabin encryption
        start = time.time()
        rabin_cipher = self._rabin_encrypt(content)
        rabin_time = time.time() - start
        self.performance.record('Rabin_Encrypt', rabin_time, len(content))
        
        # RSA key encryption
        start = time.time()
        file_key = str(rabin_cipher)[:32]
        cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[owner]['public'])
        encrypted_key = cipher_rsa.encrypt(file_key.encode())
        rsa_time = time.time() - start
        self.performance.record('RSA_Encrypt', rsa_time)
        
        file_rec = {
            'id': len(self.files) + 1,
            'owner': owner,
            'filename': filename,
            'rabin_cipher': rabin_cipher,
            'encrypted_key': base64.b64encode(encrypted_key).decode(),
            'hash': file_hash,
            'size': len(content)
        }
        
        self.files.append(file_rec)
        
        print(f"\n✓ File uploaded!")
        print(f"  MD5 hash: {md5_time:.6f}s")
        print(f"  Rabin encrypt: {rabin_time:.6f}s")
        print(f"  RSA key encrypt: {rsa_time:.6f}s")
        print(f"  Total: {md5_time + rabin_time + rsa_time:.6f}s")
        
        return file_rec['id']
    
    def download_file(self, user, file_id):
        """Download and decrypt file"""
        file_rec = next((f for f in self.files if f['id'] == file_id), None)
        if not file_rec or file_rec['owner'] != user:
            print("File not found or access denied!")
            return None
        
        # RSA key decryption
        start = time.time()
        cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[user]['private'])
        file_key = cipher_rsa.decrypt(base64.b64decode(file_rec['encrypted_key'])).decode()
        rsa_time = time.time() - start
        self.performance.record('RSA_Decrypt', rsa_time)
        
        # Rabin decryption
        start = time.time()
        content = self._rabin_decrypt(file_rec['rabin_cipher'])
        rabin_time = time.time() - start
        self.performance.record('Rabin_Decrypt', rabin_time)
        
        # MD5 verification
        start = time.time()
        computed_hash = MD5.new(content.encode()).hexdigest()
        verified = computed_hash == file_rec['hash']
        md5_time = time.time() - start
        self.performance.record('MD5_Verify', md5_time)
        
        print(f"\n✓ File downloaded!")
        print(f"  RSA key decrypt: {rsa_time:.6f}s")
        print(f"  Rabin decrypt: {rabin_time:.6f}s")
        print(f"  MD5 verify: {md5_time:.6f}s")
        print(f"  Hash verified: {verified}")
        print(f"\n  Content: {content}")
        
        return content

def menu_cloud_system():
    """Menu for cloud storage"""
    system = CloudStorageSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 3: CLOUD STORAGE (Rabin + RSA + MD5)")
        print("="*70)
        print("1. Register User")
        print("2. Upload File")
        print("3. Download File")
        print("4. View All Files")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            user = input("User ID: ")
            system.register_user(user)
        elif choice == '2':
            owner = input("Your ID: ")
            filename = input("Filename: ")
            content = input("Content: ")
            system.upload_file(owner, filename, content)
        elif choice == '3':
            user = input("Your ID: ")
            file_id = int(input("File ID: "))
            system.download_file(user, file_id)
        elif choice == '4':
            for f in system.files:
                print(f"\n[{f['id']}] {f['filename']} - Owner: {f['owner']} ({f['size']} bytes)")
        elif choice == '5':
            system.performance.print_graph("Cloud Storage Performance")
        elif choice == '6':
            system.performance.compare_algorithms('Rabin_Encrypt', 'RSA_Encrypt', 'MD5_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 4: Triple DES + ElGamal + SHA-1 (Legacy Banking)
# ═══════════════════════════════════════════════════════════════════════════

class LegacyBankingSystem:
    """Legacy system with 3DES, ElGamal, SHA-1"""
    
    def __init__(self):
        self.des3_key = b"TWENTYFOUR_BYTE_KEY!"[:24]  # 24 bytes for 3DES
        self.elgamal_keys = {}
        self.transactions = []
        self.performance = PerformanceTracker()
    
    def register_customer(self, cust_id):
        """Generate ElGamal keys"""
        start = time.time()
        bits = 512
        while True:
            q = number.getPrime(bits - 1)
            p = 2 * q + 1
            if number.isPrime(p):
                break
        while True:
            g = random.randint(2, p - 2)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                break
        x = random.randint(2, p - 2)
        y = pow(g, x, p)
        
        self.elgamal_keys[cust_id] = {'p': p, 'g': g, 'x': x, 'y': y}
        self.performance.record('ElGamal_KeyGen', time.time() - start)
        print(f"✓ Customer {cust_id} registered")
    
    def create_transaction(self, from_id, to_id, amount):
        """Create 3DES encrypted transaction"""
        if from_id not in self.elgamal_keys:
            print("Sender not registered!")
            return None
            
        txn_data = f"{from_id}:{to_id}:{amount}"
        
        # 3DES encryption
        start = time.time()
        cipher = DES3.new(self.des3_key, DES3.MODE_ECB)
        encrypted = cipher.encrypt(pad(txn_data.encode(), DES3.block_size))
        des3_time = time.time() - start
        self.performance.record('3DES_Encrypt', des3_time, len(txn_data))
        
        # SHA-1 hash
        start = time.time()
        txn_hash = SHA1.new(txn_data.encode()).hexdigest()
        sha1_time = time.time() - start
        self.performance.record('SHA1_Hash', sha1_time)
        
        # ElGamal signature
        start = time.time()
        keys = self.elgamal_keys[from_id]
        p, g, x = keys['p'], keys['g'], keys['x']
        h_int = int(txn_hash, 16) % (p - 1)
        
        while True:
            k = random.randint(2, p - 2)
            if gcd(k, p - 1) == 1:
                break
        
        r = pow(g, k, p)
        k_inv = number.inverse(k, p - 1)
        s = (k_inv * (h_int - x * r)) % (p - 1)
        elg_time = time.time() - start
        self.performance.record('ElGamal_Sign', elg_time)
        
        txn = {
            'id': len(self.transactions) + 1,
            'sender': from_id,
            'encrypted': base64.b64encode(encrypted).decode(),
            'hash': txn_hash,
            'sig_r': r,
            'sig_s': s
        }
        
        self.transactions.append(txn)
        
        print(f"\n✓ Transaction created!")
        print(f"  3DES: {des3_time:.6f}s | ElGamal: {elg_time:.6f}s | SHA-1: {sha1_time:.6f}s")
        return txn['id']
    
    def verify_transaction(self, txn_id):
        """Verify transaction"""
        txn = next((t for t in self.transactions if t['id'] == txn_id), None)
        if not txn:
            print("Transaction not found!")
            return False
        
        customer_id = txn['sender']
        if customer_id not in self.elgamal_keys:
            print("Sender's keys not found!")
            return False

        # 3DES decryption
        start = time.time()
        cipher = DES3.new(self.des3_key, DES3.MODE_ECB)
        decrypted = unpad(cipher.decrypt(base64.b64decode(txn['encrypted'])), DES3.block_size).decode()
        des3_time = time.time() - start
        self.performance.record('3DES_Decrypt', des3_time)
        
        # ElGamal verification
        start = time.time()
        keys = self.elgamal_keys[customer_id]
        p, g, y = keys['p'], keys['g'], keys['y']
        
        # Re-hash the decrypted data to check integrity
        rehash = SHA1.new(decrypted.encode()).hexdigest()
        if rehash != txn['hash']:
            print("  Hash mismatch! Data has been tampered with.")
            
        h_int = int(txn['hash'], 16) % (p - 1)
        
        v1 = (pow(y, txn['sig_r'], p) * pow(txn['sig_r'], txn['sig_s'], p)) % p
        v2 = pow(g, h_int, p)
        verified = v1 == v2
        elg_time = time.time() - start
        self.performance.record('ElGamal_Verify', elg_time)
        
        print(f"\n✓ Verification complete!")
        print(f"  3DES decrypt: {des3_time:.6f}s | ElGamal verify: {elg_time:.6f}s")
        print(f"  Data: {decrypted} | Signature: {'VALID' if verified else 'INVALID'}")
        
        return verified

def menu_legacy_banking():
    """Menu for legacy banking"""
    system = LegacyBankingSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 4: LEGACY BANKING (3DES + ElGamal + SHA-1)")
        print("="*70)
        print("1. Register Customer")
        print("2. Create Transaction")
        print("3. Verify Transaction")
        print("4. View All Transactions")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            cust = input("Customer ID: ")
            system.register_customer(cust)
        elif choice == '2':
            from_id = input("From: ")
            to_id = input("To: ")
            amount = float(input("Amount: "))
            system.create_transaction(from_id, to_id, amount)
        elif choice == '3':
            txn_id = int(input("Transaction ID: "))
            system.verify_transaction(txn_id)
        elif choice == '4':
            for t in system.transactions:
                print(f"  [{t['id']}] Sender: {t['sender']}, Hash: {t['hash'][:16]}...")
        elif choice == '5':
            system.performance.print_graph("Legacy Banking Performance")
        elif choice == '6':
            system.performance.compare_algorithms('3DES_Encrypt', 'ElGamal_Sign', 'SHA1_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 5: AES + RSA + SHA-256 (Healthcare Records)
# ═══════════════════════════════════════════════════════════════════════════

class HealthcareSystem:
    """Medical records with AES, RSA, SHA-256"""
    
    def __init__(self):
        self.aes_key = get_random_bytes(32)
        self.rsa_keys = {}
        self.records = []
        self.performance = PerformanceTracker()
    
    def register_doctor(self, doc_id):
        """Generate RSA keys for doctor"""
        start = time.time()
        key = RSA.generate(2048)
        self.rsa_keys[doc_id] = {
            'private': key,
            'public': key.publickey()
        }
        self.performance.record('RSA_KeyGen', time.time() - start)
        print(f"✓ Doctor {doc_id} registered with RSA keys")
    
    def create_medical_record(self, patient_id, doctor_id, diagnosis):
        """Create encrypted medical record"""
        if doctor_id not in self.rsa_keys:
            print("Doctor not registered!")
            return None
        
        record_data = f"Patient:{patient_id}|Diagnosis:{diagnosis}"
        
        # AES encryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(pad(record_data.encode(), AES.block_size))
        aes_time = time.time() - start
        self.performance.record('AES_Encrypt', aes_time, len(record_data))
        
        # SHA-256 hash
        start = time.time()
        record_hash = SHA256.new(record_data.encode()).hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA256_Hash', sha_time)
        
        # RSA signature
        start = time.time()
        h = SHA256.new(record_data.encode())
        signature = pkcs1_15.new(self.rsa_keys[doctor_id]['private']).sign(h)
        rsa_time = time.time() - start
        self.performance.record('RSA_Sign', rsa_time)
        
        record = {
            'id': len(self.records) + 1,
            'patient': patient_id,
            'doctor': doctor_id,
            'encrypted': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'tag': base64.b64encode(tag).decode(),
            'hash': record_hash,
            'signature': base64.b64encode(signature).decode()
        }
        
        self.records.append(record)
        
        print(f"\n✓ Medical record created!")
        print(f"  AES: {aes_time:.6f}s | RSA: {rsa_time:.6f}s | SHA-256: {sha_time:.6f}s")
        print(f"  Total: {aes_time + rsa_time + sha_time:.6f}s")
        
        return record['id']
    
    def access_record(self, record_id, doctor_id):
        """Access and verify medical record"""
        record = next((r for r in self.records if r['id'] == record_id), None)
        if not record:
            print("Record not found!")
            return None
        
        if record['doctor'] not in self.rsa_keys:
            print("Doctor's public key not found for verification!")
            return None
            
        # AES decryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX, nonce=base64.b64decode(record['nonce']))
        decrypted = unpad(cipher.decrypt(base64.b64decode(record['encrypted'])), AES.block_size).decode()
        cipher.verify(base64.b64decode(record['tag']))
        aes_time = time.time() - start
        self.performance.record('AES_Decrypt', aes_time)
        
        # RSA signature verification
        start = time.time()
        h = SHA256.new(decrypted.encode())
        try:
            pkcs1_15.new(self.rsa_keys[record['doctor']]['public']).verify(h, base64.b64decode(record['signature']))
            verified = True
        except (ValueError, TypeError):
            verified = False
        rsa_time = time.time() - start
        self.performance.record('RSA_Verify', rsa_time)
        
        print(f"\n✓ Record accessed!")
        print(f"  AES decrypt: {aes_time:.6f}s | RSA verify: {rsa_time:.6f}s")
        print(f"  Signature: {'VALID' if verified else 'INVALID'}")
        print(f"  Data: {decrypted}")
        
        return decrypted

def menu_healthcare():
    """Menu for healthcare system"""
    system = HealthcareSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 5: HEALTHCARE (AES-256 + RSA + SHA-256)")
        print("="*70)
        print("1. Register Doctor")
        print("2. Create Medical Record")
        print("3. Access Medical Record")
        print("4. View All Records")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            doc = input("Doctor ID: ")
            system.register_doctor(doc)
        elif choice == '2':
            patient = input("Patient ID: ")
            doctor = input("Doctor ID: ")
            diagnosis = input("Diagnosis: ")
            system.create_medical_record(patient, doctor, diagnosis)
        elif choice == '3':
            rec_id = int(input("Record ID: "))
            doc = input("Your doctor ID (for verification): ")
            system.access_record(rec_id, doc)
        elif choice == '4':
            for r in system.records:
                print(f"  [{r['id']}] Patient: {r['patient']} | Doctor: {r['doctor']}")
        elif choice == '5':
            system.performance.print_graph("Healthcare System Performance")
        elif choice == '6':
            system.performance.compare_algorithms('AES_Encrypt', 'RSA_Sign', 'SHA256_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 6: DES + ElGamal + MD5 (Document Management)
# ═══════════════════════════════════════════════════════════════════════════

class DocumentManagementSystem:
    """Document system with DES, ElGamal, MD5"""
    
    def __init__(self):
        self.des_key = b"12345678"
        self.elgamal_keys = {}
        self.documents = []
        self.performance = PerformanceTracker()
    
    def register_author(self, author_id):
        """Generate ElGamal keys"""
        start = time.time()
        bits = 512
        while True:
            q = number.getPrime(bits - 1)
            p = 2 * q + 1
            if number.isPrime(p):
                break
        while True:
            g = random.randint(2, p - 2)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                break
        x = random.randint(2, p - 2)
        y = pow(g, x, p)
        
        self.elgamal_keys[author_id] = {'p': p, 'g': g, 'x': x, 'y': y}
        self.performance.record('ElGamal_KeyGen', time.time() - start)
        print(f"✓ Author {author_id} registered")
    
    def create_document(self, author_id, title, content):
        """Create signed document"""
        if author_id not in self.elgamal_keys:
            print("Author not registered!")
            return None
            
        doc_data = f"{title}||{content}"
        
        # DES encryption
        start = time.time()
        cipher = DES.new(self.des_key, DES.MODE_ECB)
        encrypted = cipher.encrypt(pad(doc_data.encode(), DES.block_size))
        des_time = time.time() - start
        self.performance.record('DES_Encrypt', des_time, len(doc_data))
        
        # MD5 hash
        start = time.time()
        doc_hash = MD5.new(doc_data.encode()).hexdigest()
        md5_time = time.time() - start
        self.performance.record('MD5_Hash', md5_time)
        
        # ElGamal signature
        start = time.time()
        keys = self.elgamal_keys[author_id]
        p, g, x = keys['p'], keys['g'], keys['x']
        h_int = int(doc_hash, 16) % (p - 1)
        
        while True:
            k = random.randint(2, p - 2)
            if gcd(k, p - 1) == 1:
                break
        
        r = pow(g, k, p)
        k_inv = number.inverse(k, p - 1)
        s = (k_inv * (h_int - x * r)) % (p - 1)
        elg_time = time.time() - start
        self.performance.record('ElGamal_Sign', elg_time)
        
        doc = {
            'id': len(self.documents) + 1,
            'author': author_id,
            'encrypted': base64.b64encode(encrypted).decode(),
            'hash': doc_hash,
            'sig_r': r,
            'sig_s': s
        }
        
        self.documents.append(doc)
        
        print(f"\n✓ Document created!")
        print(f"  DES: {des_time:.6f}s | ElGamal: {elg_time:.6f}s | MD5: {md5_time:.6f}s")
        
        return doc['id']
    
    def verify_document(self, doc_id):
        """Verify document"""
        doc = next((d for d in self.documents if d['id'] == doc_id), None)
        if not doc:
            print("Document not found!")
            return False
        
        if doc['author'] not in self.elgamal_keys:
            print("Author's keys not found for verification!")
            return False

        # DES decryption
        start = time.time()
        cipher = DES.new(self.des_key, DES.MODE_ECB)
        decrypted = unpad(cipher.decrypt(base64.b64decode(doc['encrypted'])), DES.block_size).decode()
        des_time = time.time() - start
        self.performance.record('DES_Decrypt', des_time)
        
        # ElGamal verification
        start = time.time()
        keys = self.elgamal_keys[doc['author']]
        p, g, y = keys['p'], keys['g'], keys['y']
        
        # Re-hash decrypted data to check integrity
        rehash = MD5.new(decrypted.encode()).hexdigest()
        if rehash != doc['hash']:
            print("  Hash mismatch! Data has been tampered with.")

        h_int = int(doc['hash'], 16) % (p - 1)
        
        v1 = (pow(y, doc['sig_r'], p) * pow(doc['sig_r'], doc['sig_s'], p)) % p
        v2 = pow(g, h_int, p)
        verified = v1 == v2
        elg_time = time.time() - start
        self.performance.record('ElGamal_Verify', elg_time)
        
        title, content = decrypted.split('||', 1)
        
        print(f"\n✓ Document verified!")
        print(f"  DES decrypt: {des_time:.6f}s | ElGamal verify: {elg_time:.6f}s")
        print(f"  Title: {title}")
        print(f"  Content: {content}")
        print(f"  Signature: {'VALID' if verified else 'INVALID'}")
        
        return verified

def menu_document_management():
    """Menu for document management"""
    system = DocumentManagementSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 6: DOCUMENT MANAGEMENT (DES + ElGamal + MD5)")
        print("="*70)
        print("1. Register Author")
        print("2. Create Document")
        print("3. Verify Document")
        print("4. View All Documents")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            author = input("Author ID: ")
            system.register_author(author)
        elif choice == '2':
            author = input("Author ID: ")
            title = input("Title: ")
            content = input("Content: ")
            system.create_document(author, title, content)
        elif choice == '3':
            doc_id = int(input("Document ID: "))
            system.verify_document(doc_id)
        elif choice == '4':
            for d in system.documents:
                print(f"  [{d['id']}] Author: {d['author']} | Hash: {d['hash'][:16]}...")
        elif choice == '5':
            system.performance.print_graph("Document Management Performance")
        elif choice == '6':
            system.performance.compare_algorithms('DES_Encrypt', 'ElGamal_Sign', 'MD5_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 7: AES + ElGamal + MD5 (Messaging Platform)
# ═══════════════════════════════════════════════════════════════════════════

class MessagingPlatform:
    """Instant messaging with AES, ElGamal, MD5"""
    
    def __init__(self):
        self.aes_key = hashlib.sha256(b"MessagingKey").digest()
        self.elgamal_keys = {}
        self.messages = []
        self.performance = PerformanceTracker()
    
    def register_user(self, user_id):
        """Generate ElGamal keys"""
        start = time.time()
        bits = 512
        while True:
            q = number.getPrime(bits - 1)
            p = 2 * q + 1
            if number.isPrime(p):
                break
        while True:
            g = random.randint(2, p - 2)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                break
        x = random.randint(2, p - 2)
        y = pow(g, x, p)
        
        self.elgamal_keys[user_id] = {'p': p, 'g': g, 'x': x, 'y': y}
        self.performance.record('ElGamal_KeyGen', time.time() - start)
        print(f"✓ User {user_id} registered")
    
    def send_message(self, from_user, to_user, text):
        """Send encrypted message"""
        if from_user not in self.elgamal_keys:
            print("Sender not registered!")
            return None

        # AES encryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(pad(text.encode(), AES.block_size))
        aes_time = time.time() - start
        self.performance.record('AES_Encrypt', aes_time, len(text))
        
        # MD5 hash
        start = time.time()
        msg_hash = MD5.new(text.encode()).hexdigest()
        md5_time = time.time() - start
        self.performance.record('MD5_Hash', md5_time)
        
        # ElGamal signature
        start = time.time()
        keys = self.elgamal_keys[from_user]
        p, g, x = keys['p'], keys['g'], keys['x']
        h_int = int(msg_hash, 16) % (p - 1)
        
        while True:
            k = random.randint(2, p - 2)
            if gcd(k, p - 1) == 1:
                break
        
        r = pow(g, k, p)
        k_inv = number.inverse(k, p - 1)
        s = (k_inv * (h_int - x * r)) % (p - 1)
        elg_time = time.time() - start
        self.performance.record('ElGamal_Sign', elg_time)
        
        msg = {
            'id': len(self.messages) + 1,
            'from': from_user,
            'to': to_user,
            'encrypted': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'tag': base64.b64encode(tag).decode(),
            'hash': msg_hash,
            'sig_r': r,
            'sig_s': s
        }
        
        self.messages.append(msg)
        
        print(f"\n✓ Message sent!")
        print(f"  AES: {aes_time:.6f}s | ElGamal: {elg_time:.6f}s | MD5: {md5_time:.6f}s")
        
        return msg['id']
    
    def read_message(self, msg_id, user_id):
        """Read and verify message"""
        msg = next((m for m in self.messages if m['id'] == msg_id and m['to'] == user_id), None)
        if not msg:
            print("Message not found!")
            return None
        
        if msg['from'] not in self.elgamal_keys:
            print("Sender's keys not found for verification!")
            return None

        # AES decryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX, nonce=base64.b64decode(msg['nonce']))
        decrypted = unpad(cipher.decrypt(base64.b64decode(msg['encrypted'])), AES.block_size).decode()
        cipher.verify(base64.b64decode(msg['tag']))
        aes_time = time.time() - start
        self.performance.record('AES_Decrypt', aes_time)
        
        # ElGamal verification
        start = time.time()
        keys = self.elgamal_keys[msg['from']]
        p, g, y = keys['p'], keys['g'], keys['y']
        
        rehash = MD5.new(decrypted.encode()).hexdigest()
        if rehash != msg['hash']:
            print("  Hash mismatch! Message may have been tampered with.")

        h_int = int(msg['hash'], 16) % (p - 1)
        
        v1 = (pow(y, msg['sig_r'], p) * pow(msg['sig_r'], msg['sig_s'], p)) % p
        v2 = pow(g, h_int, p)
        verified = v1 == v2
        elg_time = time.time() - start
        self.performance.record('ElGamal_Verify', elg_time)
        
        print(f"\n✓ Message received!")
        print(f"  From: {msg['from']}")
        print(f"  Message: {decrypted}")
        print(f"  Signature: {'VALID' if verified else 'INVALID'}")
        
        return decrypted

def menu_messaging():
    """Menu for messaging platform"""
    system = MessagingPlatform()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 7: MESSAGING (AES + ElGamal + MD5)")
        print("="*70)
        print("1. Register User")
        print("2. Send Message")
        print("3. Read Message")
        print("4. View All Messages")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            user = input("User ID: ")
            system.register_user(user)
        elif choice == '2':
            from_u = input("From: ")
            to_u = input("To: ")
            text = input("Message: ")
            system.send_message(from_u, to_u, text)
        elif choice == '3':
            msg_id = int(input("Message ID: "))
            user = input("Your ID: ")
            system.read_message(msg_id, user)
        elif choice == '4':
            for m in system.messages:
                print(f"  [{m['id']}] {m['from']} → {m['to']}")
        elif choice == '5':
            system.performance.print_graph("Messaging Platform Performance")
        elif choice == '6':
            system.performance.compare_algorithms('AES_Encrypt', 'ElGamal_Sign', 'MD5_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 8: DES + RSA + SHA-512 (Secure File Transfer)
# ═══════════════════════════════════════════════════════════════════════════

class SecureFileTransfer:
    """File transfer with DES, RSA, SHA-512"""
    
    def __init__(self):
        self.rsa_keys = {}
        self.files = []
        self.performance = PerformanceTracker()
    
    def register_user(self, user_id):
        """Generate RSA keys"""
        start = time.time()
        key = RSA.generate(2048)
        self.rsa_keys[user_id] = {
            'private': key,
            'public': key.publickey()
        }
        self.performance.record('RSA_KeyGen', time.time() - start)
        print(f"✓ User {user_id} registered")
    
    def send_file(self, sender, recipient, filename, content):
        """Send encrypted file"""
        if recipient not in self.rsa_keys:
            print("Recipient not registered!")
            return None

        # DES key generation and encryption
        start = time.time()
        des_key = get_random_bytes(8)
        cipher_des = DES.new(des_key, DES.MODE_ECB)
        encrypted = cipher_des.encrypt(pad(content.encode(), DES.block_size))
        des_time = time.time() - start
        self.performance.record('DES_Encrypt', des_time, len(content))
        
        # SHA-512 hash
        start = time.time()
        file_hash = SHA512.new(content.encode()).hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA512_Hash', sha_time)
        
        # RSA key encryption
        start = time.time()
        cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[recipient]['public'])
        encrypted_key = cipher_rsa.encrypt(des_key)
        rsa_time = time.time() - start
        self.performance.record('RSA_Encrypt', rsa_time)
        
        file_rec = {
            'id': len(self.files) + 1,
            'sender': sender,
            'recipient': recipient,
            'filename': filename,
            'encrypted': base64.b64encode(encrypted).decode(),
            'encrypted_key': base64.b64encode(encrypted_key).decode(),
            'hash': file_hash
        }
        
        self.files.append(file_rec)
        
        print(f"\n✓ File sent!")
        print(f"  DES: {des_time:.6f}s | RSA: {rsa_time:.6f}s | SHA-512: {sha_time:.6f}s")
        
        return file_rec['id']
    
    def receive_file(self, file_id, user_id):
        """Receive and decrypt file"""
        file_rec = next((f for f in self.files if f['id'] == file_id and f['recipient'] == user_id), None)
        if not file_rec:
            print("File not found!")
            return None
        
        if user_id not in self.rsa_keys:
            print("You are not registered, cannot decrypt!")
            return None
            
        # RSA key decryption
        start = time.time()
        cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[user_id]['private'])
        des_key = cipher_rsa.decrypt(base64.b64decode(file_rec['encrypted_key']))
        rsa_time = time.time() - start
        self.performance.record('RSA_Decrypt', rsa_time)
        
        # DES decryption
        start = time.time()
        cipher_des = DES.new(des_key, DES.MODE_ECB)
        decrypted = unpad(cipher_des.decrypt(base64.b64decode(file_rec['encrypted'])), DES.block_size).decode()
        des_time = time.time() - start
        self.performance.record('DES_Decrypt', des_time)
        
        # SHA-512 verification
        start = time.time()
        computed_hash = SHA512.new(decrypted.encode()).hexdigest()
        verified = computed_hash == file_rec['hash']
        sha_time = time.time() - start
        self.performance.record('SHA512_Verify', sha_time)
        
        print(f"\n✓ File received!")
        print(f"  RSA: {rsa_time:.6f}s | DES: {des_time:.6f}s | SHA-512: {sha_time:.6f}s")
        print(f"  Filename: {file_rec['filename']}")
        print(f"  Hash verified: {verified}")
        print(f"  Content: {decrypted}")
        
        return decrypted

def menu_file_transfer():
    """Menu for file transfer"""
    system = SecureFileTransfer()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 8: FILE TRANSFER (DES + RSA + SHA-512)")
        print("="*70)
        print("1. Register User")
        print("2. Send File")
        print("3. Receive File")
        print("4. View All Files")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            user = input("User ID: ")
            system.register_user(user)
        elif choice == '2':
            sender = input("From: ")
            recipient = input("To: ")
            filename = input("Filename: ")
            content = input("Content: ")
            system.send_file(sender, recipient, filename, content)
        elif choice == '3':
            file_id = int(input("File ID: "))
            user = input("Your ID: ")
            system.receive_file(file_id, user)
        elif choice == '4':
            for f in system.files:
                print(f"  [{f['id']}] {f['filename']}: {f['sender']} → {f['recipient']}")
        elif choice == '5':
            system.performance.print_graph("File Transfer Performance")
        elif choice == '6':
            system.performance.compare_algorithms('DES_Encrypt', 'RSA_Encrypt', 'SHA512_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 9: Rabin + ElGamal + SHA-256 (Digital Library)
# ═══════════════════════════════════════════════════════════════════════════

class DigitalLibrary:
    """Library system with Rabin, ElGamal, SHA-256"""
    
    def __init__(self):
        self.rabin_keys = self._init_rabin()
        self.elgamal_keys = {}
        self.books = []
        self.performance = PerformanceTracker()
    
    def _init_rabin(self):
        """Initialize Rabin keys"""
        start = time.time()
        bits = 1024
        while True:
            p = number.getPrime(bits // 2)
            if p % 4 == 3:
                break
        while True:
            q = number.getPrime(bits // 2)
            if q % 4 == 3 and q != p:
                break
        n = p * q
        self.performance.record('Rabin_KeyGen', time.time() - start)
        return {'n': n, 'p': p, 'q': q}
    
    def register_publisher(self, pub_id):
        """Generate ElGamal keys for publisher"""
        start = time.time()
        bits = 512
        while True:
            q = number.getPrime(bits - 1)
            p = 2 * q + 1
            if number.isPrime(p):
                break
        while True:
            g = random.randint(2, p - 2)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                break
        x = random.randint(2, p - 2)
        y = pow(g, x, p)
        
        self.elgamal_keys[pub_id] = {'p': p, 'g': g, 'x': x, 'y': y}
        self.performance.record('ElGamal_KeyGen', time.time() - start)
        print(f"✓ Publisher {pub_id} registered")
    
    def publish_book(self, publisher_id, title, content):
        """Publish signed and encrypted book"""
        if publisher_id not in self.elgamal_keys:
            print("Publisher not registered!")
            return None

        book_data = f"{title}:{content}"
        
        # Rabin encryption
        start = time.time()
        msg_bytes = book_data.encode()
        redundancy = b"BOOK" + len(msg_bytes).to_bytes(2, 'big')
        full_msg = redundancy + msg_bytes
        m = int.from_bytes(full_msg, 'big')
        if m >= self.rabin_keys['n']:
            print("Content too long!")
            return None
        rabin_cipher = pow(m, 2, self.rabin_keys['n'])
        rabin_time = time.time() - start
        self.performance.record('Rabin_Encrypt', rabin_time, len(content))
        
        # SHA-256 hash
        start = time.time()
        book_hash = SHA256.new(book_data.encode()).hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA256_Hash', sha_time)
        
        # ElGamal signature
        start = time.time()
        keys = self.elgamal_keys[publisher_id]
        p, g, x = keys['p'], keys['g'], keys['x']
        h = SHA256.new(book_data.encode())
        h_int = int.from_bytes(h.digest(), 'big') % (p - 1)
        
        while True:
            k = random.randint(2, p - 2)
            if gcd(k, p - 1) == 1:
                break
        
        r = pow(g, k, p)
        k_inv = number.inverse(k, p - 1)
        s = (k_inv * (h_int - x * r)) % (p - 1)
        elg_time = time.time() - start
        self.performance.record('ElGamal_Sign', elg_time)
        
        book = {
            'id': len(self.books) + 1,
            'publisher': publisher_id,
            'rabin_cipher': rabin_cipher,
            'hash': book_hash,
            'sig_r': r,
            'sig_s': s
        }
        
        self.books.append(book)
        
        print(f"\n✓ Book published!")
        print(f"  Rabin: {rabin_time:.6f}s | ElGamal: {elg_time:.6f}s | SHA-256: {sha_time:.6f}s")
        
        return book['id']
    
    def read_book(self, book_id):
        """Read and verify book"""
        book = next((b for b in self.books if b['id'] == book_id), None)
        if not book:
            print("Book not found!")
            return None
        
        if book['publisher'] not in self.elgamal_keys:
            print("Publisher's keys not found for verification!")
            return None
            
        # Rabin decryption
        start = time.time()
        p, q, n = self.rabin_keys['p'], self.rabin_keys['q'], self.rabin_keys['n']
        c = book['rabin_cipher']
        mp = pow(c, (p + 1) // 4, p)
        mq = pow(c, (q + 1) // 4, q)
        inv_p = number.inverse(p, q)
        inv_q = number.inverse(q, p)
        a = (q * inv_q) % n
        b = (p * inv_p) % n
        roots = [
            (a * mp + b * mq) % n,
            (a * mp - b * mq) % n,
            (-a * mp + b * mq) % n,
            (-a * mp - b * mq) % n
        ]
        
        decrypted = None
        for r in roots:
            try:
                msg_bytes = r.to_bytes((r.bit_length() + 7) // 8, 'big')
                if msg_bytes[:4] == b"BOOK":
                    length = int.from_bytes(msg_bytes[4:6], 'big')
                    decrypted = msg_bytes[6:6+length].decode()
                    break
            except:
                continue
        
        rabin_time = time.time() - start
        self.performance.record('Rabin_Decrypt', rabin_time)
        
        if not decrypted:
            print("Decryption failed!")
            return None
        
        # ElGamal verification
        start = time.time()
        keys = self.elgamal_keys[book['publisher']]
        p, g, y = keys['p'], keys['g'], keys['y']
        
        rehash = SHA256.new(decrypted.encode()).hexdigest()
        if rehash != book['hash']:
            print("  Hash mismatch! Book content may have been corrupted.")

        h = SHA256.new(decrypted.encode())
        h_int = int.from_bytes(h.digest(), 'big') % (p - 1)
        
        v1 = (pow(y, book['sig_r'], p) * pow(book['sig_r'], book['sig_s'], p)) % p
        v2 = pow(g, h_int, p)
        verified = v1 == v2
        elg_time = time.time() - start
        self.performance.record('ElGamal_Verify', elg_time)
        
        title, content = decrypted.split(':', 1)
        
        print(f"\n✓ Book retrieved!")
        print(f"  Rabin: {rabin_time:.6f}s | ElGamal: {elg_time:.6f}s")
        print(f"  Title: {title}")
        print(f"  Content: {content}")
        print(f"  Signature: {'VALID' if verified else 'INVALID'}")
        
        return decrypted

def menu_digital_library():
    """Menu for digital library"""
    system = DigitalLibrary()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 9: DIGITAL LIBRARY (Rabin + ElGamal + SHA-256)")
        print("="*70)
        print("1. Register Publisher")
        print("2. Publish Book")
        print("3. Read Book")
        print("4. View All Books")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            pub = input("Publisher ID: ")
            system.register_publisher(pub)
        elif choice == '2':
            pub = input("Publisher ID: ")
            title = input("Title: ")
            content = input("Content: ")
            system.publish_book(pub, title, content)
        elif choice == '3':
            book_id = int(input("Book ID: "))
            system.read_book(book_id)
        elif choice == '4':
            for b in system.books:
                print(f"  [{b['id']}] Publisher: {b['publisher']}")
        elif choice == '5':
            system.performance.print_graph("Digital Library Performance")
        elif choice == '6':
            system.performance.compare_algorithms('Rabin_Encrypt', 'ElGamal_Sign', 'SHA256_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 10: AES + RSA + SHA-512 (Secure Chat)
# ═══════════════════════════════════════════════════════════════════════════

class SecureChatSystem:
    """Secure Chat with AES, RSA, SHA-512"""
    
    def __init__(self):
        self.aes_key = get_random_bytes(32)
        self.rsa_keys = {}
        self.messages = []
        self.performance = PerformanceTracker()
    
    def register_user(self, user_id):
        """Generate RSA keys for user"""
        start = time.time()
        key = RSA.generate(2048)
        self.rsa_keys[user_id] = {
            'private': key,
            'public': key.publickey()
        }
        self.performance.record('RSA_KeyGen', time.time() - start)
        print(f"✓ User {user_id} registered with RSA keys")
    
    def send_message(self, sender_id, message_text):
        """Create encrypted and signed message"""
        if sender_id not in self.rsa_keys:
            print("User not registered!")
            return None
        
        # AES encryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(pad(message_text.encode(), AES.block_size))
        aes_time = time.time() - start
        self.performance.record('AES_Encrypt', aes_time, len(message_text))
        
        # SHA-512 hash
        start = time.time()
        message_hash = SHA512.new(message_text.encode()).hexdigest()
        sha_time = time.time() - start
        self.performance.record('SHA512_Hash', sha_time)
        
        # RSA signature
        start = time.time()
        h = SHA512.new(message_text.encode())
        signature = pkcs1_15.new(self.rsa_keys[sender_id]['private']).sign(h)
        rsa_time = time.time() - start
        self.performance.record('RSA_Sign', rsa_time)
        
        message = {
            'id': len(self.messages) + 1,
            'sender': sender_id,
            'encrypted': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'tag': base64.b64encode(tag).decode(),
            'hash': message_hash,
            'signature': base64.b64encode(signature).decode()
        }
        
        self.messages.append(message)
        
        print(f"\n✓ Message sent!")
        print(f"  AES: {aes_time:.6f}s | RSA: {rsa_time:.6f}s | SHA-512: {sha_time:.6f}s")
        print(f"  Total: {aes_time + rsa_time + sha_time:.6f}s")
        
        return message['id']
    
    def read_message(self, message_id):
        """Access and verify medical record"""
        message = next((r for r in self.messages if r['id'] == message_id), None)
        if not message:
            print("Message not found!")
            return None
        
        if message['sender'] not in self.rsa_keys:
            print("Sender's public key not found for verification!")
            return None
            
        # AES decryption
        start = time.time()
        cipher = AES.new(self.aes_key, AES.MODE_EAX, nonce=base64.b64decode(message['nonce']))
        decrypted = unpad(cipher.decrypt(base64.b64decode(message['encrypted'])), AES.block_size).decode()
        cipher.verify(base64.b64decode(message['tag']))
        aes_time = time.time() - start
        self.performance.record('AES_Decrypt', aes_time)
        
        # RSA signature verification
        start = time.time()
        h = SHA512.new(decrypted.encode())
        try:
            pkcs1_15.new(self.rsa_keys[message['sender']]['public']).verify(h, base64.b64decode(message['signature']))
            verified = True
        except (ValueError, TypeError):
            verified = False
        rsa_time = time.time() - start
        self.performance.record('RSA_Verify', rsa_time)
        
        print(f"\n✓ Message read!")
        print(f"  Sender: {message['sender']}")
        print(f"  AES decrypt: {aes_time:.6f}s | RSA verify: {rsa_time:.6f}s")
        print(f"  Signature: {'VALID' if verified else 'INVALID'}")
        print(f"  Message: {decrypted}")
        
        return decrypted

def menu_secure_chat():
    """Menu for healthcare system"""
    system = SecureChatSystem()
    
    while True:
        print(f"\n{'='*70}")
        print("  SYSTEM 10: SECURE CHAT (AES-256 + RSA + SHA-512)")
        print("="*70)
        print("1. Register User")
        print("2. Send Message")
        print("3. Read Message")
        print("4. View All Messages")
        print("5. Performance Analysis")
        print("6. Algorithm Comparison")
        print("7. Back to Main Menu")
        print("-"*70)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            user = input("User ID: ")
            system.register_user(user)
        elif choice == '2':
            sender = input("Your User ID: ")
            message = input("Message: ")
            system.send_message(sender, message)
        elif choice == '3':
            msg_id = int(input("Message ID: "))
            system.read_message(msg_id)
        elif choice == '4':
            for m in system.messages:
                print(f"  [{m['id']}] Sender: {m['sender']}")
        elif choice == '5':
            system.performance.print_graph("Secure Chat System Performance")
        elif choice == '6':
            system.performance.compare_algorithms('AES_Encrypt', 'RSA_Sign', 'SHA512_Hash')
        elif choice == '7':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 11: Paillier + ElGamal + SHA-256 (E-Voting)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_PAILLIER:
    class SecureVotingSystem:
        """E-Voting with Paillier homomorphic encryption, ElGamal signatures, and SHA-256 receipts"""
        
        def __init__(self):
            print("Initializing Paillier keypair for election... (this may take a second)")
            start = time.time()
            self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)
            paillier_time = time.time() - start
            
            self.elgamal_keys = {}
            self.candidates = {}
            self.votes = []
            self.voter_hashes = set()
            self.performance = PerformanceTracker()
            self.performance.record('Paillier_KeyGen', paillier_time)
            print(f"✓ Paillier system ready ({paillier_time:.2f}s)")
        
        def register_voter(self, voter_id):
            """Register voter with ElGamal keys"""
            start = time.time()
            bits = 512
            while True:
                q = number.getPrime(bits - 1)
                p = 2 * q + 1
                if number.isPrime(p):
                    break
            while True:
                g = random.randint(2, p - 2)
                if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                    break
            x = random.randint(2, p - 2)
            y = pow(g, x, p)
            
            self.elgamal_keys[voter_id] = {'p': p, 'g': g, 'x': x, 'y': y}
            self.performance.record('ElGamal_KeyGen', time.time() - start)
            print(f"✓ Voter {voter_id} registered with ElGamal keys")

        def register_candidate(self, candidate_name):
            cid = len(self.candidates) + 1
            self.candidates[cid] = {
                'name': candidate_name,
                'encrypted_tally': self.public_key.encrypt(0)
            }
            print(f"✓ Candidate {candidate_name} registered with ID {cid}")
            return cid
        
        def cast_vote(self, voter_id, candidate_id):
            """Cast a signed, encrypted vote"""
            if voter_id not in self.elgamal_keys:
                print("Voter not registered!")
                return None
            if candidate_id not in self.candidates:
                print("Candidate not found!")
                return None
            if voter_id in self.voter_hashes:
                print("Voter has already voted!")
                return None
            
            vote_data = f"VOTE|{voter_id}|{candidate_id}|{datetime.now().isoformat()}"
            
            # SHA-256 hash (Receipt)
            start = time.time()
            vote_hash = SHA256.new(vote_data.encode()).digest()
            hash_hex = vote_hash.hex()
            sha_time = time.time() - start
            self.performance.record('SHA256_Hash', sha_time)
            
            # ElGamal signature
            start = time.time()
            keys = self.elgamal_keys[voter_id]
            p, g, x = keys['p'], keys['g'], keys['x']
            h_int = int.from_bytes(vote_hash, 'big') % (p - 1)
            
            while True:
                k = random.randint(2, p - 2)
                if gcd(k, p - 1) == 1:
                    break
            
            r = pow(g, k, p)
            k_inv = number.inverse(k, p - 1)
            s = (k_inv * (h_int - x * r)) % (p - 1)
            elg_time = time.time() - start
            self.performance.record('ElGamal_Sign', elg_time)
            
            # Paillier encryption
            start = time.time()
            encrypted_vote = self.public_key.encrypt(1)
            paillier_time = time.time() - start
            self.performance.record('Paillier_Encrypt', paillier_time)
            
            # Homomorphic Addition to Tally
            self.candidates[candidate_id]['encrypted_tally'] += encrypted_vote
            
            vote_record = {
                'voter_id': voter_id,
                'candidate_id': candidate_id,
                'encrypted_vote': encrypted_vote,
                'hash': hash_hex,
                'sig_r': r,
                'sig_s': s
            }
            self.votes.append(vote_record)
            self.voter_hashes.add(voter_id)
            
            print(f"\n✓ Vote cast successfully!")
            print(f"  Paillier: {paillier_time:.6f}s | ElGamal: {elg_time:.6f}s | SHA-256: {sha_time:.6f}s")
            print(f"  Your Receipt Hash: {hash_hex}")
            return hash_hex

        def tally_results(self):
            """Tally and verify all votes"""
            print(f"\n{'='*70}")
            print("  ELECTION TALLY & VERIFICATION")
            print("="*70)
            
            total_verified = 0
            for vote in self.votes:
                # ElGamal/SHA-256 verification
                start = time.time()
                keys = self.elgamal_keys[vote['voter_id']]
                p, g, y = keys['p'], keys['g'], keys['y']
                h_int = int.from_bytes(bytes.fromhex(vote['hash']), 'big') % (p - 1)
                
                v1 = (pow(y, vote['sig_r'], p) * pow(vote['sig_r'], vote['sig_s'], p)) % p
                v2 = pow(g, h_int, p)
                if v1 == v2:
                    total_verified += 1
                self.performance.record('ElGamal_Verify', time.time() - start)

            print(f"  Total Votes: {len(self.votes)}")
            print(f"  Verified Signatures: {total_verified}")
            
            if total_verified != len(self.votes):
                print("  WARNING: Mismatch in vote count and verified signatures!")
            
            print("\nFinal Results:")
            results = []
            for cid, data in self.candidates.items():
                start = time.time()
                count = self.private_key.decrypt(data['encrypted_tally'])
                self.performance.record('Paillier_Decrypt', time.time() - start)
                results.append((data['name'], count))
                
            for name, count in sorted(results, key=lambda x: x[1], reverse=True):
                print(f"  - {name:20s}: {count} votes")
            print("="*70)

    def menu_voting_system():
        system = SecureVotingSystem()
        
        while True:
            print(f"\n{'='*70}")
            print("  SYSTEM 11: E-VOTING (Paillier + ElGamal + SHA-256)")
            print("="*70)
            print("1. Register Voter (Generates ElGamal keys)")
            print("2. Register Candidate")
            print("3. Cast Vote (Paillier Encrypt + ElGamal Sign + SHA-256 Hash)")
            print("4. Tally Results (Paillier Decrypt + Verify All Signatures)")
            print("5. View All Votes (Encrypted)")
            print("6. Performance Analysis")
            print("7. Algorithm Comparison")
            print("8. Back to Main Menu")
            print("-"*70)
            
            choice = input("Choice: ").strip()
            
            if choice == '1':
                voter_id = input("Voter ID: ")
                system.register_voter(voter_id)
            elif choice == '2':
                name = input("Candidate Name: ")
                system.register_candidate(name)
            elif choice == '3':
                voter_id = input("Your Voter ID: ")
                cid = int(input("Candidate ID to vote for: "))
                system.cast_vote(voter_id, cid)
            elif choice == '4':
                system.tally_results()
            elif choice == '5':
                for v in system.votes:
                    print(f"  Voter: {v['voter_id']} | Hash: {v['hash'][:16]}...")
            elif choice == '6':
                system.performance.print_graph("E-Voting System Performance")
            elif choice == '7':
                system.performance.compare_algorithms('Paillier_Encrypt', 'ElGamal_Sign', 'SHA256_Hash')
            elif choice == '8':
                break
else:
    def menu_voting_system():
        print("\nPaillier (phe) library not installed. System 11 is disabled.")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 12: Hill + RSA + SHA-256 (Classical-Modern Hybrid)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_NUMPY:
    class HillHybridSystem:
        """Secure message container with Hill, RSA, and SHA-256"""
        
        def __init__(self):
            self.rsa_keys = {}
            self.messages = []
            self.performance = PerformanceTracker()
        
        def register_user(self, user_id):
            """Generate RSA keys"""
            start = time.time()
            key = RSA.generate(2048)
            self.rsa_keys[user_id] = {
                'private': key,
                'public': key.publickey()
            }
            self.performance.record('RSA_KeyGen', time.time() - start)
            print(f"✓ User {user_id} registered")

        def _hill_encrypt(self, plaintext, key_matrix):
            """Internal Hill encrypt"""
            n = key_matrix.shape[0]
            plaintext = plaintext.upper().replace(" ", "")
            original_len = len(plaintext)
            while len(plaintext) % n != 0:
                plaintext += 'X'
            
            ciphertext = ""
            for i in range(0, len(plaintext), n):
                block = np.array([ord(c) - 65 for c in plaintext[i:i+n]])
                encrypted_block = np.dot(key_matrix, block) % 26
                ciphertext += "".join([chr(c + 65) for c in encrypted_block])
            return ciphertext, original_len

        def _hill_decrypt(self, ciphertext, key_matrix, original_len):
            """Internal Hill decrypt"""
            n = key_matrix.shape[0]
            inv_matrix = matrix_mod_inv(key_matrix, 26)
            
            plaintext = ""
            for i in range(0, len(ciphertext), n):
                block = np.array([ord(c) - 65 for c in ciphertext[i:i+n]])
                decrypted_block = np.dot(inv_matrix, block) % 26
                plaintext += "".join([chr(c + 65) for c in decrypted_block])
            return plaintext[:original_len] # Return only original message
            
        def send_message(self, sender, recipient, message, key_matrix_str):
            """Encrypt message with Hill, key with RSA, and hash with SHA-256"""
            if recipient not in self.rsa_keys:
                print("Recipient not found!")
                return None
            
            try:
                key_values = [int(v) for v in key_matrix_str.split()]
                n = int(np.sqrt(len(key_values)))
                if n*n != len(key_values):
                    raise ValueError("Key values do not form a square matrix")
                key_matrix = np.array(key_values).reshape((n, n))
                # Test invertibility
                matrix_mod_inv(key_matrix, 26)
            except Exception as e:
                print(f"Invalid key matrix: {e}")
                return None

            # SHA-256 hash
            start = time.time()
            msg_hash = SHA256.new(message.encode()).hexdigest()
            sha_time = time.time() - start
            self.performance.record('SHA256_Hash', sha_time)
            
            # Hill encryption
            start = time.time()
            encrypted_msg, original_len = self._hill_encrypt(message, key_matrix)
            hill_time = time.time() - start
            self.performance.record('Hill_Encrypt', hill_time, len(message))
            
            # RSA key encryption
            start = time.time()
            # We encrypt the key string AND the original length
            key_and_len = f"{key_matrix_str}|{original_len}"
            cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[recipient]['public'])
            encrypted_key = cipher_rsa.encrypt(key_and_len.encode())
            rsa_time = time.time() - start
            self.performance.record('RSA_Encrypt', rsa_time)
            
            msg = {
                'id': len(self.messages) + 1,
                'sender': sender,
                'recipient': recipient,
                'encrypted_msg': encrypted_msg,
                'encrypted_key': base64.b64encode(encrypted_key).decode(),
                'hash': msg_hash
            }
            self.messages.append(msg)
            
            print(f"\n✓ Message sent!")
            print(f"  Hill Encrypt: {hill_time:.6f}s")
            print(f"  RSA Key Encrypt: {rsa_time:.6f}s")
            print(f"  SHA-256 Hash: {sha_time:.6f}s")
            return msg['id']

        def read_message(self, user_id, msg_id):
            """Decrypt key with RSA, message with Hill, and verify with SHA-256"""
            msg = next((m for m in self.messages if m['id'] == msg_id and m['recipient'] == user_id), None)
            if not msg:
                print("Message not found or access denied!")
                return None

            # RSA key decryption
            start = time.time()
            cipher_rsa = PKCS1_OAEP.new(self.rsa_keys[user_id]['private'])
            key_and_len = cipher_rsa.decrypt(base64.b64decode(msg['encrypted_key'])).decode()
            key_matrix_str, original_len = key_and_len.split('|')
            original_len = int(original_len)
            rsa_time = time.time() - start
            self.performance.record('RSA_Decrypt', rsa_time)
            
            key_values = [int(v) for v in key_matrix_str.split()]
            n = int(np.sqrt(len(key_values)))
            key_matrix = np.array(key_values).reshape((n, n))
            
            # Hill decryption
            start = time.time()
            decrypted_msg = self._hill_decrypt(msg['encrypted_msg'], key_matrix, original_len)
            hill_time = time.time() - start
            self.performance.record('Hill_Decrypt', hill_time)

            # SHA-256 verification
            start = time.time()
            computed_hash = SHA256.new(decrypted_msg.encode()).hexdigest()
            verified = (computed_hash == msg['hash'])
            sha_time = time.time() - start
            self.performance.record('SHA256_Verify', sha_time)
            
            print(f"\n✓ Message received!")
            print(f"  RSA Key Decrypt: {rsa_time:.6f}s")
            print(f"  Hill Decrypt: {hill_time:.6f}s")
            print(f"  SHA-256 Verify: {sha_time:.6f}s")
            print(f"  Hash verified: {verified}")
            print(f"  Message: {decrypted_msg}")
            
            return decrypted_msg

    def menu_hill_hybrid():
        system = HillHybridSystem()
        
        while True:
            print(f"\n{'='*70}")
            print("  SYSTEM 12: HYBRID (Hill Cipher + RSA + SHA-256)")
            print("="*70)
            print("1. Register User (Get RSA keys)")
            print("2. Send Encrypted Message")
            print("3. Read Message")
            print("4. View All Messages")
            print("5. Performance Analysis")
            print("6. Algorithm Comparison")
            print("7. Back to Main Menu")
            print("-"*70)
            
            choice = input("Choice: ").strip()
            
            if choice == '1':
                user = input("User ID: ")
                system.register_user(user)
            elif choice == '2':
                sender = input("From: ")
                recipient = input("To: ")
                message = input("Message: ")
                key_str = input("Hill Key (e.g., '3 3 2 7' for 2x2): ")
                system.send_message(sender, recipient, message, key_str)
            elif choice == '3':
                user = input("Your ID: ")
                msg_id = int(input("Message ID: "))
                system.read_message(user, msg_id)
            elif choice == '4':
                for m in system.messages:
                    print(f"  [{m['id']}] {m['sender']} → {m['recipient']}: {m['encrypted_msg'][:20]}...")
            elif choice == '5':
                system.performance.print_graph("Hill Hybrid System Performance")
            elif choice == '6':
                system.performance.compare_algorithms('Hill_Encrypt', 'RSA_Encrypt', 'SHA256_Hash')
            elif choice == '7':
                break
else:
    def menu_hill_hybrid():
        print("\nNumPy not installed. System 12 is disabled.")


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 2: CLASSICAL CIPHER & LAB SCENARIOS
# (From PDF files)
#
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 13: Hill Cipher Lab (Standalone)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_NUMPY:
    class HillCipherSystem:
        def __init__(self):
            self.messages = []
        
        def hill_encrypt(self, plaintext, key_matrix):
            n = len(key_matrix)
            plaintext = plaintext.upper().replace(' ', '')
            while len(plaintext) % n != 0:
                plaintext += 'X'
            
            result = ""
            for i in range(0, len(plaintext), n):
                block = [ord(c) - 65 for c in plaintext[i:i+n]]
                encrypted = np.dot(key_matrix, block) % 26
                result += ''.join(chr(int(c) + 65) for c in encrypted)
            return result

        def hill_decrypt(self, ciphertext, key_matrix):
            key_inv = matrix_mod_inv(key_matrix, 26)
            return self.hill_encrypt(ciphertext, key_inv)

    def menu_hill_cipher_lab():
        system = HillCipherSystem()
        
        matrices = {
            '1': ('[[3,3], [2,7]]', np.array([[3,3],[2,7]])),
        }
        
        while True:
            print("\n" + "="*60)
            print("  SYSTEM 13: HILL CIPHER LAB")
            print("="*60)
            print("1. Encrypt with Matrix [[3,3],[2,7]]")
            print("2. Decrypt with Matrix [[3,3],[2,7]]")
            print("3. Custom Matrix Encryption")
            print("4. Test Matrix Invertibility")
            print("5. View Encryption History")
            print("6. Back to Main Menu")
            print("-"*60)
            choice = input("Choice: ").strip()
            
            if choice == '1':
                plaintext = input("Enter plaintext: ")
                matrix = np.array([[3,3],[2,7]])
                ciphertext = system.hill_encrypt(plaintext, matrix)
                print(f"\nPlaintext: {plaintext}")
                print(f"Matrix:\n[[3,3],[2,7]]")
                print(f"Ciphertext: {ciphertext}")
                system.messages.append({
                    'plaintext': plaintext,
                    'ciphertext': ciphertext,
                    'matrix': matrix
                })
            elif choice == '2':
                ciphertext = input("Enter ciphertext: ")
                matrix = np.array([[3,3],[2,7]])
                try:
                    plaintext = system.hill_decrypt(ciphertext, matrix)
                    print(f"\nCiphertext: {ciphertext}")
                    print(f"Plaintext: {plaintext}")
                except Exception as e:
                    print(f"Error: {e}")
            elif choice == '3':
                try:
                    size = int(input("Matrix size (2 for 2x2, 3 for 3x3): "))
                    print(f"Enter {size}x{size} matrix values:")
                    matrix = []
                    for i in range(size):
                        row = input(f"Row {i+1} (space-separated): ").split()
                        matrix.append([int(x) for x in row])
                    matrix = np.array(matrix)
                    plaintext = input("Plaintext: ")
                    ciphertext = system.hill_encrypt(plaintext, matrix)
                    print(f"Ciphertext: {ciphertext}")
                except Exception as e:
                    print(f"Error: {e}")
            elif choice == '4':
                try:
                    size = int(input("Matrix size: "))
                    print(f"Enter {size}x{size} matrix:")
                    matrix = []
                    for i in range(size):
                        row = input(f"Row {i+1}: ").split()
                        matrix.append([int(x) for x in row])
                    matrix = np.array(matrix)
                    det = int(np.round(np.linalg.det(matrix)))
                    det_mod = det % 26
                    print(f"\nDeterminant: {det}")
                    print(f"Det mod 26: {det_mod}")
                    if gcd(det_mod, 26) == 1:
                        inv = mod_inverse(det_mod, 26)
                        print(f"✓ Matrix is INVERTIBLE")
                        print(f"  Inverse of {det_mod} mod 26 = {inv}")
                    else:
                        print(f"✗ Matrix is NOT INVERTIBLE")
                        print(f"  gcd({det_mod}, 26) = {gcd(det_mod, 26)}")
                except Exception as e:
                    print(f"Error: {e}")
            elif choice == '5':
                if not system.messages:
                    print("No messages!")
                else:
                    for i, msg in enumerate(system.messages, 1):
                        print(f"\n--- Message {i} ---")
                        print(f"Plaintext: {msg['plaintext']}")
                        print(f"Ciphertext: {msg['ciphertext']}")
            elif choice == '6':
                break
else:
    def menu_hill_cipher_lab():
        print("\nNumPy not installed. System 13 is disabled.")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 14: Vigenère Cipher Lab (Standalone)
# ═══════════════════════════════════════════════════════════════════════════

class VigenereLab:
    def __init__(self):
        self.history = []

    def vigenere_encrypt(self, plaintext, key):
        result = ""
        key = key.upper()
        key_index = 0
        for char in plaintext.upper():
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - 65
                result += chr((ord(char) - 65 + shift) % 26 + 65)
                key_index += 1
            else:
                result += char
        return result

    def vigenere_decrypt(self, ciphertext, key):
        result = ""
        key = key.upper()
        key_index = 0
        for char in ciphertext.upper():
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - 65
                result += chr((ord(char) - 65 - shift) % 26 + 65)
                key_index += 1
            else:
                result += char
        return result
    
    def show_detailed_encryption(self, plaintext, key):
        plaintext = plaintext.upper().replace(" ", "")
        key = key.upper()
        print(f"\n=== DETAILED VIGENÈRE ENCRYPTION ===")
        print(f"Key: {key}")
        print(f"Plaintext: {plaintext}\n")
        result = ""
        key_index = 0
        print("Step-by-step:")
        for i, char in enumerate(plaintext):
            if char.isalpha():
                key_char = key[key_index % len(key)]
                shift = ord(key_char) - 65
                plain_num = ord(char) - 65
                cipher_num = (plain_num + shift) % 26
                cipher_char = chr(cipher_num + 65)
                print(f"  {char} + {key_char}  →  {plain_num:2d} + {shift:2d} = {cipher_num:2d}  →  {cipher_char}")
                result += cipher_char
                key_index += 1
        print(f"\nCiphertext: {result}")
        return result

def menu_vigenere_lab():
    system = VigenereLab()
    while True:
        print("\n" + "="*60)
        print("  SYSTEM 14: VIGENÈRE CIPHER LAB")
        print("="*60)
        print("1. Encrypt with key 'DOLLARS'")
        print("2. Encrypt with key 'HEALTH'")
        print("3. Detailed Encryption (Step-by-step)")
        print("4. Decrypt Message")
        print("5. Custom Key Encryption")
        print("6. Back to Main Menu")
        print("-"*60)
        choice = input("Choice: ").strip()

        if choice == '1' or choice == '2':
            key = 'DOLLARS' if choice == '1' else 'HEALTH'
            plaintext = input("Enter plaintext: ")
            ciphertext = system.vigenere_encrypt(plaintext, key)
            print(f"\nKey: {key}")
            print(f"Plaintext: {plaintext}")
            print(f"Ciphertext: {ciphertext}")
            system.history.append((key, plaintext, ciphertext))
        elif choice == '3':
            plaintext = input("Enter plaintext: ")
            key = input("Enter key: ")
            system.show_detailed_encryption(plaintext, key)
        elif choice == '4':
            ciphertext = input("Enter ciphertext: ")
            key = input("Enter key: ")
            plaintext = system.vigenere_decrypt(ciphertext, key)
            print(f"\nKey: {key}")
            print(f"Ciphertext: {ciphertext}")
            print(f"Plaintext: {plaintext}")
        elif choice == '5':
            plaintext = input("Enter plaintext: ")
            key = input("Enter key: ")
            ciphertext = system.vigenere_encrypt(plaintext, key)
            print(f"\nKey: {key}")
            print(f"Plaintext: {plaintext}")
            print(f"Ciphertext: {ciphertext}")
        elif choice == '6':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 15: Playfair Cipher Lab (Standalone)
# ═══════════════════════════════════════════════════════════════════════════

class PlayfairLab:
    def __init__(self):
        self.messages = []

    def create_matrix(self, key):
        key = key.upper().replace('J', 'I').replace(' ', '')
        matrix = []
        used = set()
        for char in key:
            if char.isalpha() and char not in used:
                matrix.append(char)
                used.add(char)
        for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':
            if char not in used:
                matrix.append(char)
        return [matrix[i:i+5] for i in range(0, 25, 5)]

    def display_matrix(self, matrix):
        print("\n Playfair Matrix:")
        print(" +" + "-"*11 + "+")
        for row in matrix:
            print(" | " + " ".join(row) + " |")
        print(" +" + "-"*11 + "+")

    def find_position(self, matrix, char):
        for i, row in enumerate(matrix):
            for j, c in enumerate(row):
                if c == char:
                    return i, j
        return None

    def prepare_digraphs(self, text):
        text = text.upper().replace('J', 'I').replace(' ', '')
        text = "".join(filter(str.isalpha, text))
        digraphs = []
        i = 0
        while i < len(text):
            a = text[i]
            b = text[i+1] if i+1 < len(text) else 'X'
            if a == b:
                digraphs.append((a, 'X'))
                i += 1
            else:
                digraphs.append((a, b))
                i += 2
        return digraphs

    def playfair_encrypt(self, plaintext, key):
        matrix = self.create_matrix(key)
        digraphs = self.prepare_digraphs(plaintext)
        result = ""
        for a, b in digraphs:
            r1, c1 = self.find_position(matrix, a)
            r2, c2 = self.find_position(matrix, b)
            if r1 == r2: # Same row
                result += matrix[r1][(c1 + 1) % 5]
                result += matrix[r2][(c2 + 1) % 5]
            elif c1 == c2: # Same column
                result += matrix[(r1 + 1) % 5][c1]
                result += matrix[(r2 + 1) % 5][c2]
            else: # Rectangle
                result += matrix[r1][c2]
                result += matrix[r2][c1]
        return result, digraphs

    def playfair_decrypt(self, ciphertext, key):
        matrix = self.create_matrix(key)
        result = ""
        for i in range(0, len(ciphertext), 2):
            if i+1 >= len(ciphertext):
                break
            a, b = ciphertext[i], ciphertext[i+1]
            r1, c1 = self.find_position(matrix, a)
            r2, c2 = self.find_position(matrix, b)
            if r1 == r2:
                result += matrix[r1][(c1 - 1) % 5]
                result += matrix[r2][(c2 - 1) % 5]
            elif c1 == c2:
                result += matrix[(r1 - 1) % 5][c1]
                result += matrix[(r2 - 1) % 5][c2]
            else:
                result += matrix[r1][c2]
                result += matrix[r2][c1]
        return result

def menu_playfair_lab():
    system = PlayfairLab()
    while True:
        print("\n" + "="*60)
        print("  SYSTEM 15: PLAYFAIR CIPHER LAB")
        print("="*60)
        print("1. Encrypt with key 'GUIDANCE'")
        print("2. View 'GUIDANCE' Matrix")
        print("3. Detailed Encryption (show digraphs)")
        print("4. Decrypt Message")
        print("5. Custom Key Encryption")
        print("6. Back to Main Menu")
        print("-"*60)
        choice = input("Choice: ").strip()

        if choice == '1':
            plaintext = input("Enter plaintext: ")
            ciphertext, digraphs = system.playfair_encrypt(plaintext, 'GUIDANCE')
            print(f"\nKey: GUIDANCE")
            print(f"Plaintext: {plaintext}")
            print(f"Digraphs: {' '.join([f'{a}{b}' for a,b in digraphs])}")
            print(f"Ciphertext: {ciphertext}")
        elif choice == '2':
            matrix = system.create_matrix('GUIDANCE')
            system.display_matrix(matrix)
            print("\n Note: J is replaced with I")
        elif choice == '3':
            plaintext = input("Enter plaintext: ")
            key = input("Enter key (default GUIDANCE): ") or 'GUIDANCE'
            matrix = system.create_matrix(key)
            system.display_matrix(matrix)
            digraphs = system.prepare_digraphs(plaintext)
            print(f"\nDigraphs: {digraphs}")
            result = ""
            print("\nEncryption steps:")
            for a, b in digraphs:
                r1, c1 = system.find_position(matrix, a)
                r2, c2 = system.find_position(matrix, b)
                rule = ""
                if r1 == r2:
                    enc_a, enc_b = matrix[r1][(c1+1)%5], matrix[r2][(c2+1)%5]
                    rule = "same row"
                elif c1 == c2:
                    enc_a, enc_b = matrix[(r1+1)%5][c1], matrix[(r2+1)%5][c2]
                    rule = "same column"
                else:
                    enc_a, enc_b = matrix[r1][c2], matrix[r2][c1]
                    rule = "rectangle"
                result += enc_a + enc_b
                print(f"  {a}{b}  →  {enc_a}{enc_b}  ({rule})")
            print(f"\nFinal ciphertext: {result}")
        elif choice == '4':
            ciphertext = input("Enter ciphertext: ")
            key = input("Enter key (default GUIDANCE): ") or 'GUIDANCE'
            plaintext = system.playfair_decrypt(ciphertext, key)
            print(f"\nKey: {key}")
            print(f"Ciphertext: {ciphertext}")
            print(f"Plaintext: {plaintext}")
        elif choice == '5':
            plaintext = input("Enter plaintext: ")
            key = input("Enter key: ")
            ciphertext, _ = system.playfair_encrypt(plaintext, key)
            print(f"\nKey: {key}")
            print(f"Plaintext: {plaintext}")
            print(f"Ciphertext: {ciphertext}")
        elif choice == '6':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 16: Affine Cipher Brute Force Lab
# ═══════════════════════════════════════════════════════════════════════════

class AffineBruteForce:
    def __init__(self):
        self.results = []

    def affine_encrypt(self, text, a, b):
        result = ""
        for char in text.upper():
            if char.isalpha():
                result += chr((a * (ord(char) - 65) + b) % 26 + 65)
            else:
                result += char
        return result

    def brute_force_affine(self, known_plain, known_cipher):
        known_plain = known_plain.upper()
        known_cipher = known_cipher.upper()
        valid_keys = []
        valid_a = [a for a in range(1, 26) if gcd(a, 26) == 1]
        
        print(f"\nSearching for keys...")
        print(f"Plaintext: {known_plain}")
        print(f"Ciphertext: {known_cipher}")
        print(f"\nValid 'a' values: {valid_a}")
        print(f"Testing {len(valid_a)} x 26 = {len(valid_a) * 26} combinations...\n")
        
        for a in valid_a:
            for b in range(26):
                test = self.affine_encrypt(known_plain, a, b)
                if test == known_cipher:
                    valid_keys.append((a, b))
                    print(f"  ✓ FOUND: a={a}, b={b}")
                    test_plain = "THEQUICKBROWNFOX"
                    test_cipher = self.affine_encrypt(test_plain, a, b)
                    print(f"    Test: '{test_plain}' → '{test_cipher}'")
        
        if not valid_keys:
            print("No valid keys found!")
        else:
            print(f"\nTotal keys found: {len(valid_keys)}")
        self.results = valid_keys
        return valid_keys

def menu_affine_brute():
    system = AffineBruteForce()
    while True:
        print("\n" + "="*60)
        print("  SYSTEM 16: AFFINE CIPHER BRUTE FORCE ATTACK")
        print("="*60)
        print("1. Brute Force with Known Plaintext-Ciphertext")
        print("2. Example: 'ab' → 'GL'")
        print("3. Example: 'HELLO' → 'RCLLA'")
        print("4. Test Found Keys")
        print("5. Back to Main Menu")
        print("-"*60)
        choice = input("Choice: ").strip()

        if choice == '1':
            plain = input("Known plaintext: ")
            cipher = input("Known ciphertext: ")
            system.brute_force_affine(plain, cipher)
        elif choice == '2':
            print("\n=== EXAMPLE FROM LAB MANUAL ===")
            system.brute_force_affine("ab", "GL")
        elif choice == '3':
            print("\n=== EXAMPLE 2 ===")
            system.brute_force_affine("HELLO", "RCLLA")
        elif choice == '4':
            if not system.results:
                print("No keys found yet! Run brute force first.")
            else:
                test_text = input("Test plaintext: ")
                print("\nTesting all found keys:")
                for a, b in system.results:
                    cipher = system.affine_encrypt(test_text, a, b)
                    print(f"  Key ({a}, {b}): {test_text} → {cipher}")
        elif choice == '5':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 17: Dual Layer (Additive + Vigenère)
# ═══════════════════════════════════════════════════════════════════════════

class DualCipherSystem:
    def __init__(self):
        self.messages = []

    def additive_encrypt(self, text, key):
        result = ""
        for char in text.upper():
            if char.isalpha():
                result += chr((ord(char) - 65 + key) % 26 + 65)
            else:
                result += char
        return result

    def additive_decrypt(self, text, key):
        return self.additive_encrypt(text, -key)

    def vigenere_encrypt(self, text, key):
        result = ""
        key = key.upper()
        key_index = 0
        for char in text.upper():
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - 65
                result += chr((ord(char) - 65 + shift) % 26 + 65)
                key_index += 1
            else:
                result += char
        return result

    def vigenere_decrypt(self, text, key):
        result = ""
        key = key.upper()
        key_index = 0
        for char in text.upper():
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - 65
                result += chr((ord(char) - 65 - shift) % 26 + 65)
                key_index += 1
            else:
                result += char
        return result

    def double_encrypt(self, plaintext, additive_key, vigenere_key):
        """First apply Additive, then Vigenère"""
        step1 = self.additive_encrypt(plaintext, additive_key)
        step2 = self.vigenere_encrypt(step1, vigenere_key)
        return step1, step2

    def double_decrypt(self, ciphertext, additive_key, vigenere_key):
        """Reverse order: Vigenère first, then Additive"""
        step1 = self.vigenere_decrypt(ciphertext, vigenere_key)
        step2 = self.additive_decrypt(step1, additive_key)
        return step1, step2

def menu_dual_cipher():
    system = DualCipherSystem()
    while True:
        print("\n" + "="*60)
        print("  SYSTEM 17: DUAL CIPHER (Additive + Vigenère)")
        print("="*60)
        print("1. Encrypt Message (Double Layer)")
        print("2. Decrypt Message (Double Layer)")
        print("3. View Encryption Steps")
        print("4. Back to Main Menu")
        print("-"*60)
        choice = input("Enter choice: ").strip()

        if choice == '1':
            plaintext = input("Enter plaintext: ")
            add_key = int(input("Enter Additive key (0-25): "))
            vig_key = input("Enter Vigenère key: ")
            intermediate, final = system.double_encrypt(plaintext, add_key, vig_key)
            print("\n Encryption Complete!")
            print(f"  Original: {plaintext}")
            print(f"  After Additive ({add_key}): {intermediate}")
            print(f"  After Vigenère ('{vig_key}'): {final}")
            system.messages.append({
                'plaintext': plaintext, 'additive_key': add_key,
                'vigenere_key': vig_key, 'intermediate': intermediate,
                'ciphertext': final
            })
        elif choice == '2':
            ciphertext = input("Enter ciphertext: ")
            add_key = int(input("Enter Additive key: "))
            vig_key = input("Enter Vigenère key: ")
            intermediate, plaintext = system.double_decrypt(ciphertext, add_key, vig_key)
            print("\n Decryption Complete!")
            print(f"  Ciphertext: {ciphertext}")
            print(f"  After Vigenère decryption: {intermediate}")
            print(f"  Final Plaintext: {plaintext}")
        elif choice == '3':
            if not system.messages:
                print("No messages stored!")
            else:
                for i, msg in enumerate(system.messages, 1):
                    print(f"\n--- Message {i} ---")
                    print(f"  Plaintext: {msg['plaintext']}")
                    print(f"  Keys: Add={msg['additive_key']}, Vig='{msg['vigenere_key']}'")
                    print(f"  Step 1: {msg['intermediate']}")
                    print(f"  Final: {msg['ciphertext']}")
        elif choice == '4':
            break

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 18: Dual Layer (Affine + Playfair)
# ═══════════════════════════════════════════════════════════════════════════

class AffinePlayfairSystem:
    def __init__(self):
        self.documents = []
        self.playfair_util = PlayfairLab() # Reuse Playfair logic

    def affine_encrypt(self, text, a, b):
        if gcd(a, 26) != 1:
            raise ValueError(f"Key 'a'={a} is not coprime with 26")
        result = ""
        for char in text.upper():
            if char.isalpha():
                result += chr((a * (ord(char) - 65) + b) % 26 + 65)
            else:
                result += char
        return result

    def affine_decrypt(self, text, a, b):
        a_inv = mod_inverse(a, 26)
        if a_inv is None:
            raise ValueError(f"Key 'a'={a} is not invertible")
        result = ""
        for char in text.upper():
            if char.isalpha():
                result += chr((a_inv * (ord(char) - 65 - b)) % 26 + 65)
            else:
                result += char
        return result

    def double_encrypt(self, content, a, b, playfair_key):
        step1 = self.affine_encrypt(content, a, b)
        step2, _ = self.playfair_util.playfair_encrypt(step1, playfair_key)
        return step1, step2

    def double_decrypt(self, ciphertext, a, b, playfair_key):
        step1 = self.playfair_util.playfair_decrypt(ciphertext, playfair_key)
        step2 = self.affine_decrypt(step1, a, b)
        return step1, step2

def menu_affine_playfair():
    system = AffinePlayfairSystem()
    while True:
        print("\n" + "="*60)
        print("  SYSTEM 18: AFFINE + PLAYFAIR CIPHER SYSTEM")
        print("="*60)
        print("1. Encrypt Document")
        print("2. Decrypt Document")
        print("3. View All Documents")
        print("4. Test Affine Key Validity")
        print("5. Display Playfair Matrix")
        print("6. Back to Main Menu")
        print("-"*60)
        choice = input("Enter choice: ").strip()

        if choice == '1':
            doc_id = input("Document ID: ")
            content = input("Document content: ")
            try:
                a = int(input("Affine key 'a': "))
                b = int(input("Affine key 'b': "))
                playfair_key = input("Playfair key: ")
                step1, step2 = system.double_encrypt(content, a, b, playfair_key)
                system.documents.append({
                    'id': doc_id, 'original': content,
                    'affine_keys': (a, b), 'playfair_key': playfair_key,
                    'after_affine': step1, 'final_cipher': step2
                })
                print(f"\n Document '{doc_id}' encrypted!")
                print(f"  After Affine: {step1}")
                print(f"  After Playfair: {step2}")
            except Exception as e:
                print(f"ERROR: {e}")
        elif choice == '2':
            doc_id = input("Document ID: ")
            doc = next((d for d in system.documents if d['id'] == doc_id), None)
            if doc:
                try:
                    a, b = doc['affine_keys']
                    playfair_key = doc['playfair_key']
                    step1, step2 = system.double_decrypt(doc['final_cipher'], a, b, playfair_key)
                    print(f"\n Decryption of '{doc_id}':")
                    print(f"  Ciphertext: {doc['final_cipher']}")
                    print(f"  After Playfair decryption: {step1}")
                    print(f"  Final plaintext: {step2}")
                except Exception as e:
                    print(f"Decryption Error: {e}")
            else:
                print("Document not found!")
        elif choice == '3':
            if not system.documents:
                print("No documents stored!")
            else:
                for doc in system.documents:
                    print(f"\n  ID: {doc['id']}")
                    print(f"  Original: {doc['original'][:30]}...")
                    print(f"  Encrypted: {doc['final_cipher'][:30]}...")
        elif choice == '4':
            a = int(input("Enter 'a' value to test: "))
            if gcd(a, 26) == 1:
                print(f"  ✓ {a} is VALID (coprime with 26)")
                print(f"    Inverse: {mod_inverse(a, 26)}")
            else:
                print(f"  ✗ {a} is INVALID (not coprime with 26)")
        elif choice == '5':
            key = input("Enter Playfair key: ")
            matrix = system.playfair_util.create_matrix(key)
            system.playfair_util.display_matrix(matrix)
        elif choice == '6':
            break

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 3: BENCHMARK & MASTER MENU
#
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL BENCHMARK SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on all algorithms"""
    print(f"\n{'='*70}")
    print("  COMPREHENSIVE ALGORITHM BENCHMARK")
    print("="*70)
    
    test_data = "The quick brown fox jumps over the lazy dog" * 10
    results = {}
    
    print("\nBenchmarking... (this may take a moment)")
    
    # DES
    print("  Testing DES...")
    des_key = b"12345678"
    cipher = DES.new(des_key, DES.MODE_ECB)
    start = time.time()
    for _ in range(100):
        cipher.encrypt(pad(test_data[:100].encode(), DES.block_size))
    results['DES'] = (time.time() - start) / 100
    
    # AES
    print("  Testing AES-256...")
    aes_key = get_random_bytes(32)
    cipher = AES.new(aes_key, AES.MODE_EAX)
    start = time.time()
    for _ in range(100):
        cipher = AES.new(aes_key, AES.MODE_EAX)
        cipher.encrypt_and_digest(pad(test_data[:100].encode(), AES.block_size))
    results['AES-256'] = (time.time() - start) / 100
    
    # RSA
    print("  Testing RSA-2048...")
    rsa_key = RSA.generate(2048)
    cipher_rsa = PKCS1_OAEP.new(rsa_key.publickey())
    start = time.time()
    for _ in range(10):
        cipher_rsa.encrypt(test_data[:32].encode())
    results['RSA-2048'] = (time.time() - start) / 10
    
    # SHA-256
    print("  Testing SHA-256...")
    start = time.time()
    for _ in range(1000):
        SHA256.new(test_data.encode()).hexdigest()
    results['SHA-256'] = (time.time() - start) / 1000
    
    # SHA-512
    print("  Testing SHA-512...")
    start = time.time()
    for _ in range(1000):
        SHA512.new(test_data.encode()).hexdigest()
    results['SHA-512'] = (time.time() - start) / 1000
    
    # MD5
    print("  Testing MD5...")
    start = time.time()
    for _ in range(1000):
        MD5.new(test_data.encode()).hexdigest()
    results['MD5'] = (time.time() - start) / 1000
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("\nSpeed Ranking (fastest to slowest):")
    scale = 5000
    for i, (alg, t) in enumerate(sorted_results, 1):
        bar = '█' * int(t * scale)
        if int(t * scale) == 0 and t > 0:
            bar = '·'
        print(f"  {i}. {alg:15s}: {t:.8f}s per operation")
        print(f"     {bar}")
    
    print("\nRelative Performance:")
    fastest = sorted_results[0][1]
    for alg, t in sorted_results[1:]:
        ratio = t / fastest
        print(f"  {alg} is {ratio:.1f}x slower than {sorted_results[0][0]}")
    
    print("\n" + "="*70)

# ═══════════════════════════════════════════════════════════════════════════
# MASTER MENU
# ═══════════════════════════════════════════════════════════════════════════

def master_menu():
    """Master menu to select systems"""
    
    while True:
        print("\n" + "="*70)
        print("  ICT3141 ULTIMATE EXAM SCRIPT - MASTER MENU")
        print("="*70)
        print("\n--- PART 1: COMPLETE 3-ALGORITHM SYSTEMS ---")
        print("  1. Secure Email (DES + RSA + SHA-256)")
        print("  2. Banking (AES-256 + ElGamal + SHA-512)")
        print("  3. Cloud Storage (Rabin + RSA + MD5)")
        print("  4. Legacy Banking (3DES + ElGamal + SHA-1)")
        print("  5. Healthcare (AES-256 + RSA + SHA-256)")
        print("  6. Document Management (DES + ElGamal + MD5)")
        print("  7. Messaging (AES + ElGamal + MD5)")
        print("  8. File Transfer (DES + RSA + SHA-512)")
        print("  9. Digital Library (Rabin + ElGamal + SHA-256)")
        print(" 10. Secure Chat (AES-256 + RSA + SHA-512)")
        
        if HAS_PAILLIER:
            print(" 11. E-Voting (Paillier + ElGamal + SHA-256) [Paillier enabled]")
        else:
            print(" 11. E-Voting (DISABLED - 'phe' library not found)")
            
        if HAS_NUMPY:
            print(" 12. Hybrid (Hill Cipher + RSA + SHA-256) [NumPy enabled]")
        else:
            print(" 12. Hybrid (DISABLED - 'numpy' library not found)")

        print("\n--- PART 2: CLASSICAL & LAB SCENARIOS ---")
        if HAS_NUMPY:
            print(" 13. Hill Cipher Lab (Standalone)")
        else:
            print(" 13. Hill Cipher Lab (DISABLED - 'numpy' not found)")
        print(" 14. Vigenère Cipher Lab (Standalone)")
        print(" 15. Playfair Cipher Lab (Standalone)")
        print(" 16. Affine Brute Force Lab")
        print(" 17. Dual Layer (Additive + Vigenère)")
        if HAS_NUMPY:
            print(" 18. Dual Layer (Affine + Playfair)")
        else:
             print(" 18. Dual Layer (Affine + Playfair) (DISABLED - 'numpy' not found)")
        
        print("\n--- PART 3: TOOLS & EXIT ---")
        print(" 19. Universal Algorithm Benchmark")
        print(" 20. About This Script")
        print(" 21. Exit")
        print("-"*70)
        
        choice = input("\nSelect system (1-21): ").strip()
        
        if choice == '1':
            menu_email_system()
        elif choice == '2':
            menu_banking_system()
        elif choice == '3':
            menu_cloud_system()
        elif choice == '4':
            menu_legacy_banking()
        elif choice == '5':
            menu_healthcare()
        elif choice == '6':
            menu_document_management()
        elif choice == '7':
            menu_messaging()
        elif choice == '8':
            menu_file_transfer()
        elif choice == '9':
            menu_digital_library()
        elif choice == '10':
            menu_secure_chat()
        elif choice == '11':
            menu_voting_system()
        elif choice == '12':
            menu_hill_hybrid()
        elif choice == '13':
            menu_hill_cipher_lab()
        elif choice == '14':
            menu_vigenere_lab()
        elif choice == '15':
            menu_playfair_lab()
        elif choice == '16':
            menu_affine_brute()
        elif choice == '17':
            menu_dual_cipher()
        elif choice == '18':
            if HAS_NUMPY:
                menu_affine_playfair()
            else:
                print("\nNumPy not installed. This system is disabled.")
        elif choice == '19':
            run_comprehensive_benchmark()
        elif choice == '20':
            print("\n" + "="*70)
            print("  ABOUT THIS SCRIPT")
            print("="*70)
            print("\n  ICT3141 Ultimate Exam Script")
            print("  Version: 3.0 (All Scenarios Included)")
            print("  Total Systems: 18 complete scenarios")
            print("    - 12 x 3-Algorithm Complete Systems")
            print("    - 6 x Classical & Lab Scenarios")
            print("\n  All systems include:")
            print("    ✓ Complete menu-driven interface")
            print("    ✓ Performance tracking & graphs (where applicable)")
            print("    ✓ Code structured for independent use")
            print("\n  Ready for your ICT3141 exam!")
            print("="*70)
        elif choice == '21':
            print("\nGood luck with your exam! 🍀")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    try:
        master_menu()
    except KeyboardInterrupt:
        print("\n\nExiting. Good luck!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        print("Please ensure all required libraries are installed and up to date.")
        sys.exit(1)
