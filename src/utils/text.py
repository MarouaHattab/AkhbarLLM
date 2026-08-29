def contains_chinese_characters(text: str) -> bool:
    return any(
        "\u4e00" <= character <= "\u9fff"
        or "\u3400" <= character <= "\u4dbf"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )

