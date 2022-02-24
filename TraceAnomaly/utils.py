def get_unique_service_sequence(service_sequence: dict):
    result = ""
    unique_sequence = set(service_sequence)
    for v in sorted(unique_sequence):
        result = result + v + " "
    return result