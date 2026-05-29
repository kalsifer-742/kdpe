def process(email):
    email["content"] = email["content"][:1000]
    return email