def format_email(email):
    return (f"From: {email['from']}\nTo: {email['to']}\nSubject: {email['subject']}\n{email['content']}") # limited for API limits