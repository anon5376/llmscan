"""Read large token tables in blocks along either axis from checkpoint storage."""

import numpy as np


class _TokenTable:
    def __init__(self, checkpoint, name, dtype, rows_per_block):
        self.checkpoint = checkpoint
        self.name = name
        self.dtype = dtype
        self.tokens, self.features = checkpoint.tensors[name].shape
        self.rows_per_block = rows_per_block
        self.strip_start = 0
        self.strip = None

    def token_rows(self, rows):
        if isinstance(rows, slice):
            start, stop, _ = rows.indices(self.tokens)
            return self.checkpoint.matrix_rows(self.name, start, stop, slice(None), self.dtype)
        # Only the selected tokens enter the joint interval certificate.
        indices = np.asarray(rows)
        result = np.empty((len(indices), self.features), self.dtype)
        for i, token in enumerate(indices):
            result[i] = self.checkpoint.matrix_rows(self.name, token, token + 1, slice(None), self.dtype)[0]
        return result

    def feature_rows(self, rows):
        start, stop, _ = rows.indices(self.features)
        width = stop - start
        if width <= 0:
            return np.empty((0, self.tokens), self.dtype)
        offset = start - self.strip_start
        if self.strip is None or offset < 0 or offset + width > len(self.strip):
            # Nearby feature probes touch the same mapped pages. Bring in
            # several consecutive probe blocks per traversal of the token axis.
            capacity = min(self.features // 2, (64 << 20) // (self.tokens * self.dtype.itemsize))
            end = min(self.features, start + width * max(1, capacity // width))
            self.strip = None
            block = np.empty((self.tokens, end - start), self.dtype)
            for token in range(0, self.tokens, self.rows_per_block):
                tail = min(token + self.rows_per_block, self.tokens)
                block[token:tail] = self.checkpoint.matrix_rows(self.name, token, tail, slice(start, end), self.dtype)
            self.strip_start, self.strip = start, block.T
            offset = 0
        return self.strip[offset:offset + width]


class _BlockView:
    """The row blocks, selected columns and extrema used by the analysis kernels."""

    def __init__(self, shape, dtype, bounds, read_rows, read_columns):
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.dtype = dtype
        self.bounds = bounds
        self.read_rows = read_rows
        self.read_columns = read_columns

    def min(self):
        return self.bounds[0]

    def max(self):
        return self.bounds[1]

    def __len__(self):
        return self.shape[0]

    @property
    def T(self):
        return _BlockView(self.shape[::-1], self.dtype, self.bounds, self.read_columns, self.read_rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            if rows != slice(None):
                raise IndexError("expected a selection of complete columns")
            return self.read_columns(columns).T
        return self.read_rows(key)


def token_blocks(base, suspect, name, rows_per_block):
    """Preserve the working dtype, extrema and subtraction order of the dense scan."""
    tokens, features = base.tensors[name].shape
    wide = "F64" in (base.tensors[name].dtype, suspect.tensors[name].dtype)
    dtype = np.dtype(np.float64 if wide else np.float32)
    tables = [_TokenTable(c, name, dtype, rows_per_block) for c in (base, suspect)]
    bounds = np.tile([np.inf, -np.inf], (3, 1))
    for start in range(0, tokens, rows_per_block):
        a, b = [t.token_rows(slice(start, start + rows_per_block)) for t in tables]
        for span, block in zip(bounds, (a, b, b - a)):
            span[0] = np.minimum(span[0], block.min())
            span[1] = np.maximum(span[1], block.max())
    a, b = [_BlockView((features, tokens), dtype, span, t.feature_rows, t.token_rows)
            for t, span in zip(tables, bounds)]
    delta = _BlockView(a.shape, dtype, bounds[2],
                       lambda sl: b.read_rows(sl) - a.read_rows(sl),
                       lambda sl: b.read_columns(sl) - a.read_columns(sl))
    return a, b, delta
