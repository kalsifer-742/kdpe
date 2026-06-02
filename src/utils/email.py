def format_email(email):
    return (f"From: {email['from']}\nTo: {email['to']}\nSubject: {email['subject']}\n{email['content'][:500]}...") # limited for API limits