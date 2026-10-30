# BSD 2-Clause License
# Code with intentional issues for review: SQL injection, no error handling, style issues.

def divide(a, b):
    return a / b


def process_items(items):
    result = []
    for i in range(len(items)):
        result.append(items[i] * 2)
    return result


def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query
