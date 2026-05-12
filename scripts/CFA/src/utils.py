def validate_positive_number(value):
    if value <= 0:
        raise ValueError("Value must be a positive number.")
    return value

def validate_percentage(value):
    if not (0 <= value <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    return value

def format_currency(value):
    return "${:,.2f}".format(value)

def format_percentage(value):
    return "{:.2f}%".format(value)