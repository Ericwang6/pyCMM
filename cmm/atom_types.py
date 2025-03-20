from typing import List

def get_expected_connectivities(type_names: List[str]):
    connectivities = {
        "O_water": 2,
        "H_water": 1,
    }
    all_connectivities = [connectivities[name] if name in connectivities else 0 for name in type_names]
    return all_connectivities