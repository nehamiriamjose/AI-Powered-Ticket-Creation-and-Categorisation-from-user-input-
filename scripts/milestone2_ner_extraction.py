import re

def extract_entities(text):
    entities = {}

    # Username pattern
    usernames = re.findall(r'\b[a-zA-Z0-9_]{3,}\b', text)

    # Machine names (PC-101, SERVER_12)
    machines = re.findall(r'\b(?:PC|SERVER|LAPTOP)[-_]?\d+\b', text, re.IGNORECASE)

    # Error codes (ERR_404, 0x80070005)
    error_codes = re.findall(r'\b(ERR[_-]?\d+|0x[a-fA-F0-9]+)\b', text)

    entities['usernames'] = list(set(usernames))
    entities['machines'] = list(set(machines))
    entities['error_codes'] = list(set(error_codes))

    return entities


# Example test
ticket = "User john_doe cannot login to SERVER_12 due to ERR_403"
print(extract_entities(ticket))
