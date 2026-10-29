import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LaplaceQuadrature:
    """Laplace approximation for 1 / x on a positive interval."""

    exponents: np.ndarray
    weights: np.ndarray
    ymin: float
    ymax: float
    errmax: float = np.nan
    source: str = ""

    def __post_init__(self):
        exponents = np.asarray(self.exponents, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if exponents.ndim != 1 or weights.ndim != 1:
            raise ValueError("exponents and weights must be one-dimensional")
        if exponents.shape != weights.shape:
            raise ValueError("exponents and weights must have the same shape")
        if exponents.size == 0:
            raise ValueError("at least one quadrature point is required")
        if self.ymin <= 0.0:
            raise ValueError("ymin must be positive")
        if self.ymax < self.ymin:
            raise ValueError("ymax must be greater than or equal to ymin")
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "weights", weights)

    @property
    def nlap(self):
        return self.exponents.size

    def approximate_inverse(self, x):
        x = np.asarray(x, dtype=float)
        return np.einsum(
            "k,k...->...",
            self.weights,
            np.exp(-self.exponents[:, None] * x.reshape(1, -1)),
        ).reshape(x.shape)

    def validate(self, ngrid=10000):
        grid = np.geomspace(self.ymin, self.ymax, int(ngrid))
        exact = 1.0 / grid
        approx = self.approximate_inverse(grid)
        abs_err = np.abs(approx - exact)
        rel_err = abs_err / np.abs(exact)
        return {
            "max_abs": float(np.max(abs_err)),
            "max_rel": float(np.max(rel_err)),
            "rms_abs": float(np.sqrt(np.mean(abs_err**2))),
            "rms_rel": float(np.sqrt(np.mean(rel_err**2))),
        }

    def save(self, filename):
        np.savez(
            filename,
            exponents=self.exponents,
            weights=self.weights,
            ymin=self.ymin,
            ymax=self.ymax,
            errmax=self.errmax,
            source=self.source,
        )


def load(filename):
    with np.load(filename, allow_pickle=False) as data:
        source = ""
        if "source" in data:
            source = str(data["source"])
        exponents = np.array(data["exponents"], copy=True)
        weights = np.array(data["weights"], copy=True)
        ymin = float(data["ymin"])
        ymax = float(data["ymax"])
        errmax = float(data["errmax"]) if "errmax" in data else np.nan
        return LaplaceQuadrature(
            exponents,
            weights,
            ymin,
            ymax,
            errmax,
            source,
        )


def from_reference_interval(exponents, weights, ymin, ymax, source=""):
    """Scale a [1, ymax / ymin] approximation to [ymin, ymax]."""

    return LaplaceQuadrature(
        np.asarray(exponents, dtype=float) / ymin,
        np.asarray(weights, dtype=float) / ymin,
        float(ymin),
        float(ymax),
        source=source,
    )


def parse_laplace_minimax_output(text):
    """Parse stdout produced by the Fortran laplace_minimax test/driver."""

    ymin = np.nan
    ymax = np.nan
    errmax = np.nan

    range_match = re.search(
        r"range or orbital energy denominator:\s*"
        r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
        text,
    )
    if range_match:
        ymin = _fortran_float(range_match.group(1))
        ymax = _fortran_float(range_match.group(2))

    err_match = re.search(
        r"maximum absolute error of distribution:\s*([-+0-9.eEdD]+)", text
    )
    if err_match:
        errmax = abs(_fortran_float(err_match.group(1)))

    rows = []
    in_table = False
    for line in text.splitlines():
        if "exponents" in line and "weights" in line:
            in_table = True
            continue
        if not in_table:
            continue
        match = re.match(
            r"\s*\d+\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s*$", line
        )
        if match:
            rows.append((_fortran_float(match.group(1)), _fortran_float(match.group(2))))
        elif rows and line.strip() == "":
            break

    if not rows:
        raise ValueError("No exponents/weights table found in laplace_minimax output")

    exponents, weights = np.array(rows, dtype=float).T
    return LaplaceQuadrature(exponents, weights, ymin, ymax, errmax, "laplace-minimax")


def from_laplace_minimax_output(filename):
    return parse_laplace_minimax_output(Path(filename).read_text())


def from_init_table(root, ymin, ymax, nlap):
    """Load pretabulated laplace-minimax initial parameters and scale them."""

    root = Path(root)
    entries = _read_init_para(root / "data" / "init_para.txt")
    ratio = float(ymax) / float(ymin)
    entry = _select_entry(entries, int(nlap), ratio)
    quad = from_reference_interval(
        entry["weights_exponents"][1],
        entry["weights_exponents"][0],
        ymin,
        ymax,
        source=f"laplace-minimax:init_para:{entry['token']}",
    )
    object.__setattr__(quad, "errmax", _read_init_error(root, int(nlap), ratio))
    return quad


def from_denominators(root, denominators, nlap):
    den = np.asarray(denominators, dtype=float)
    if np.any(den <= 0.0):
        raise ValueError("all denominators must be positive")
    return from_init_table(root, float(np.min(den)), float(np.max(den)), nlap)


def _read_init_para(filename):
    entries = []
    token = None
    weights = []
    exponents = []
    token_re = re.compile(r"^1_xk(\d{2})_(\S+)")
    value_re = re.compile(r"^\s*([-+0-9.eEdD]+)\s+\{(omega|alpha)\s+\d+\s*\}")

    for line in Path(filename).read_text().splitlines():
        token_match = token_re.match(line)
        if token_match:
            if token is not None:
                entries.append(_make_entry(token, nlap, range_code, weights, exponents))
            nlap = int(token_match.group(1))
            range_code = token_match.group(2)
            token = token_match.group(0)
            weights = []
            exponents = []
            continue

        value_match = value_re.match(line)
        if value_match is None:
            continue
        value = _fortran_float(value_match.group(1))
        if value_match.group(2) == "omega":
            weights.append(value)
        else:
            exponents.append(value)

    if token is not None:
        entries.append(_make_entry(token, nlap, range_code, weights, exponents))
    return entries


def _make_entry(token, nlap, range_code, weights, exponents):
    if len(weights) != nlap or len(exponents) != nlap:
        raise ValueError(f"incomplete quadrature table for {token}")
    return {
        "token": token,
        "nlap": nlap,
        "range": _range_code_to_float(range_code),
        "weights_exponents": (
            np.array(weights, dtype=float),
            np.array(exponents, dtype=float),
        ),
    }


def _select_entry(entries, nlap, ratio):
    candidates = [entry for entry in entries if entry["nlap"] == nlap]
    if not candidates:
        raise ValueError(f"no laplace-minimax table for nlap={nlap}")
    lower = candidates[0]
    for entry in candidates:
        if entry["range"] <= ratio:
            lower = entry
            continue
        return lower if lower["range"] <= ratio else entry
    return lower


def _read_init_error(root, nlap, ratio):
    filename = Path(root) / "data" / "init_error.txt"
    if not filename.exists():
        return np.nan
    selected = None
    token_re = re.compile(r"^1_xk(\d{2})_(\S+)\s+([-+0-9.eEdD]+)")
    for line in filename.read_text().splitlines():
        match = token_re.match(line)
        if match is None or int(match.group(1)) != nlap:
            continue
        err = abs(_fortran_float(match.group(3)))
        entry_range = _range_code_to_float(match.group(2))
        if entry_range <= ratio:
            selected = err
            continue
        return selected if selected is not None else err
    return np.nan if selected is None else selected


def _range_code_to_float(code):
    if "E" in code:
        mantissa, exponent = code.split("E", 1)
        return float(mantissa) * 10.0 ** int(exponent)
    return float(code[0]) + float(code[1:]) / 1000.0


def _fortran_float(value):
    return float(value.replace("D", "E").replace("d", "e"))
