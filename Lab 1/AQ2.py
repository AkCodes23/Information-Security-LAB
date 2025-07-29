def vigenere_cipher(text, key):
    text, key = text.lower(), key.lower()
    result = ''
    for i in range(len(text)):
        t = ord(text[i]) - ord('a')
        k = ord(key[i % len(key)]) - ord('a')
        result += chr((t + k) % 26 + ord('a'))
    return result

message = "lifeisfullofsurprises".replace(" ", "").lower()
keyword = "HEALTH"
encrypted = vigenere_cipher(message, keyword)
print("Vigenère Encrypted:", encrypted)
