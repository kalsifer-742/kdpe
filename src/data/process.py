def process(email):
    email["content"] = email["content"][:1000].strip()
    return email