def custom_hash(input_string: str) -> int:
    # Initial hash value
    hash_value = 5381

    for char in input_string:
        # Multiply by 33 and add ASCII value of the character
        hash_value = (hash_value * 33) + ord(char)

        # Bitwise mixing: XOR with shifted value for better distribution
        hash_value ^= (hash_value >> 13)
        hash_value ^= (hash_value << 7)

    # Keep hash value within 32-bit unsigned integer range
    hash_value &= 0xFFFFFFFF

    return hash_value


# Example usage
if __name__ == "__main__":
    test_str = "Hello, World!"
    print(f"Hash for '{test_str}': {custom_hash(test_str)}")
