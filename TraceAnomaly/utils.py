def get_unique_service_sequence(service_sequence: dict):
    service_count = {}
    for v in service_sequence:
        if service_count.get(v) is None:
            service_count[v] = 0
        service_count[v] += 1
    result = ""
    unique_sequence = set(service_sequence)
    for v in sorted(unique_sequence):
        result = result + v + str(service_count[v]) + " "
    return result