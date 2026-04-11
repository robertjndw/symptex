def _overlap_suffix_prefix_len(text: str, delimiter: str) -> int:
    """Length of longest suffix of text that is also a prefix of delimiter."""
    max_prefix_len = min(len(text), len(delimiter) - 1)
    for prefix_len in range(max_prefix_len, 0, -1):
        if text.endswith(delimiter[:prefix_len]):
            return prefix_len
    return 0


class StreamDelimitedBlockFilter:
    """
    Incrementally removes all text enclosed by start/end delimiters from a stream.
    Works correctly when delimiters are split across arbitrary chunk boundaries.
    """

    def __init__(self, start_delimiter: str, end_delimiter: str) -> None:
        self._start_delimiter = start_delimiter
        self._end_delimiter = end_delimiter
        self._inside_block = False
        self._pending = ""

    def consume(self, chunk_text: str) -> str:
        if not chunk_text:
            return ""

        text = self._pending + chunk_text
        self._pending = ""
        emitted_parts: list[str] = []
        index = 0

        while index < len(text):
            if self._inside_block:
                end_index = text.find(self._end_delimiter, index)
                if end_index == -1:
                    hidden_tail = text[index:]
                    keep_len = _overlap_suffix_prefix_len(hidden_tail, self._end_delimiter)
                    self._pending = hidden_tail[-keep_len:] if keep_len else ""
                    return "".join(emitted_parts)

                index = end_index + len(self._end_delimiter)
                self._inside_block = False
                continue

            start_index = text.find(self._start_delimiter, index)
            if start_index == -1:
                visible_tail = text[index:]
                keep_len = _overlap_suffix_prefix_len(visible_tail, self._start_delimiter)
                if keep_len:
                    emitted_parts.append(visible_tail[:-keep_len])
                    self._pending = visible_tail[-keep_len:]
                else:
                    emitted_parts.append(visible_tail)
                return "".join(emitted_parts)

            emitted_parts.append(text[index:start_index])
            index = start_index + len(self._start_delimiter)
            self._inside_block = True

        return "".join(emitted_parts)

    def flush(self) -> str:
        if self._inside_block:
            self._pending = ""
            return ""

        trailing_text = self._pending
        self._pending = ""
        return trailing_text
