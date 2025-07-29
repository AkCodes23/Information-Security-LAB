def modinv(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

def additive_cipher(text, key, mode='encrypt'):
    shift = key if mode == 'encrypt' else -key
    return ''.join(chr((ord(c) - 97 + shift) % 26 + 97) for c in text if c.isalpha())

def multiplicative_cipher(text, key, mode='encrypt'):
    inv = modinv(key, 26)
    if mode == 'decrypt' and inv is None:
        raise ValueError(f"Key {key} has no modular inverse mod 26. Choose a coprime key.")
    result = ''
    multiplier = key if mode == 'encrypt' else inv
    for c in text:
        if c.isalpha():
            num = ord(c) - 97
            result += chr((num * multiplier) % 26 + 97)
    return result

def affine_cipher(text, a, b, mode='encrypt'):
    a_inv = modinv(a, 26)
    result = ''
    for c in text:
        if c.isalpha():
            x = ord(c) - 97
            if mode == 'encrypt':
                result += chr((a * x + b) % 26 + 97)
            else:
                result += chr((a_inv * (x - b)) % 26 + 97)
    return result

def main():
    original = "iamlearninginformationsecurity"
    while True:
        print("\nCipher Menu")
        print("1. Additive Cipher")
        print("2. Multiplicative Cipher")
        print("3. Affine Cipher")
        print("4. Exit")
        choice = input("Choose an option (1-4): ")

        if choice == '1':
            key = int(input("Enter Key:"))
            enc = additive_cipher(original, key)
            dec = additive_cipher(enc, key, 'decrypt')
            print("\nAdditive Cipher\nEncrypted:", enc)
            print("Decrypted:", dec)
        elif choice == '2':
            key = int(input("Enter Key:"))
            enc = multiplicative_cipher(original, key)
            dec = multiplicative_cipher(enc, key, 'decrypt')
            print("\nMultiplicative Cipher\nEncrypted:", enc)
            print("Decrypted:", dec)
        elif choice == '3':
            key1 = int(input("Enter Multiplicative Key:"))
            key2 = int(input("Enter Additive Key:"))
            enc = affine_cipher(original, key1, key2)
            dec = affine_cipher(enc, key1, key2, 'decrypt')
            print("\nAffine Cipher\nEncrypted:", enc)
            print("Decrypted:", dec)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Try again!")

if __name__ == "__main__":
    main()
