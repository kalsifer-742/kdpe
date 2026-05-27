import random

def format_emails(emails):
    formatted = []
    for email in emails:
        formatted.append(f"From: {email['from']}\nTo: {email['to']}\nSubject: {email['subject']}\n{email['body'][:500]}...") #TODO reading the entire mail crashes the program
    return "\n\n---\n\n".join(formatted)

def sample_emails(emails, n, seed):
    random.seed(seed)
    sampled_emails = random.sample(emails, n)
    formatted_emails = format_emails(sampled_emails)
    return formatted_emails