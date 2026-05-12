def validate_positive_number(value):
    try:
        number = float(value)
        if number <= 0:
            raise ValueError("The number must be positive.")
        return number
    except ValueError:
        raise ValueError("Invalid input. Please enter a positive numeric value.")

def validate_percentage(value):
    try:
        percentage = float(value)
        if percentage < 0 or percentage > 100:
            raise ValueError("The percentage must be between 0 and 100.")
        return percentage / 100  # Convert to decimal
    except ValueError:
        raise ValueError("Invalid input. Please enter a numeric value between 0 and 100.")

def validate_years(value):
    try:
        years = int(value)
        if years <= 0:
            raise ValueError("The number of years must be positive.")
        return years
    except ValueError:
        raise ValueError("Invalid input. Please enter a positive integer for years.")

def validate_cash_flows(value):
    try:
        cash_flows = [float(cf) for cf in value.split(",")]
        if not cash_flows:
            raise ValueError("Cash flows cannot be empty.")
        return cash_flows
    except ValueError:
        raise ValueError("Invalid input. Please enter numeric values separated by commas.")

def validate_compounding_frequency(value):
    try:
        frequency = int(value)
        if frequency <= 0:
            raise ValueError("Compounding frequency must be a positive integer.")
        return frequency
    except ValueError:
        raise ValueError("Invalid input. Please enter a positive integer for compounding frequency.")