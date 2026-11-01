"""CTC units from VieGem-ASR: 0 is blank, 1 is unknown, 2+ index Gemma IDs."""


class Units:
    def __init__(self, gemma_ids: list[int]):
        self.gemma_ids = list(gemma_ids)

    def __len__(self) -> int:
        return len(self.gemma_ids) + 2

    def decode(self, tokenizer, units: list[int]) -> str:
        return tokenizer.decode([self.gemma_ids[u - 2] for u in units if u >= 2]).strip()
