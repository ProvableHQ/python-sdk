from typing import Optional, Sequence, Tuple

def generate_abi(
    program_name: str,
    bytecode: str,
    network: str,
    imports: Optional[Sequence[Tuple[str, str]]] = None,
) -> str: ...
def check_compatibility(candidate_abi_json: str, standard_abi_json: str) -> list[str]: ...
