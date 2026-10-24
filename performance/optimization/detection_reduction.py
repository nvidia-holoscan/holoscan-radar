# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-derived, rational interval oracle for DetectionReduction native captures.

Bounds cover arbitrary addition order, separate or fused multiply/add, and
flush-to-zero. No candidate values participate in construction of the bounds.
This checks admissibility, not downstream equivalence or every computed tie.
"""

import math
import struct
from fractions import Fraction as Q

VERSION = "detection-reduction-v1"


class Unsupported(ValueError):
    """Evidence cannot be certified under this numerical model."""


class Arithmetic:
    def __init__(self, precision):
        bits, exponent = {"float": (24, -126), "double": (53, -1022)}[precision]
        self.bits = bits
        self.u = Q(1, 2**bits)
        self.tiny = Q(2) ** exponent
        self.maximum = (2 - 2 * self.u) * Q(2) ** (-exponent + 1)

    def rounded(self, lo, hi):
        if lo == hi == 0:
            return Q(0), Q(0)
        error = self.u * max(abs(lo), abs(hi)) + self.tiny
        if max(abs(lo - error), abs(hi + error)) > self.maximum:
            raise Unsupported("possible overflow")
        return lo - error, hi + error

    def add(self, a, b):
        return self.rounded(a[0] + b[0], a[1] + b[1])

    def mul(self, a, b):
        corners = [x * y for x in a for y in b]
        return self.rounded(min(corners), max(corners))

    def reciprocal(self, a):
        if a[0] <= 0:
            raise Unsupported("total weight not provably positive")
        return self.rounded(1 / a[1], 1 / a[0])

    def square(self, a):
        lo = 0 if a[0] <= 0 <= a[1] else min(x * x for x in a)
        return self.rounded(lo, max(x * x for x in a))

    def sqrt(self, interval):
        # Rational dyadic enclosures: no host floating sqrt is an oracle.
        def enclosure(x):
            if x <= 0:
                return Q(0), Q(0)
            exponent = (x.numerator.bit_length() - x.denominator.bit_length()) // 2
            quantum = Q(2) ** (exponent - self.bits - 16)
            scaled = x / (quantum * quantum)
            lower = math.isqrt(scaled.numerator // scaled.denominator) * quantum
            return lower, lower if lower * lower == x else lower + quantum

        lo = enclosure(max(Q(0), interval[0]))[0]
        hi = enclosure(max(Q(0), interval[1]))[1]
        lower, upper = self.rounded(lo, hi)
        return max(Q(0), lower), upper

    def sum(self, terms):
        # gamma_n bounds every permutation of n additions, including initial 0.
        if all(t == (0, 0) for t in terms):
            return Q(0), Q(0)
        n = len(terms)
        if n * self.u >= 1:
            raise Unsupported("accumulation bound unresolved")
        magnitude = sum(max(abs(x), abs(y)) for x, y in terms)
        if magnitude > self.maximum:
            raise Unsupported("possible intermediate accumulation overflow")
        error = (n * self.u * magnitude + n * self.tiny) / (1 - n * self.u)
        return self.rounded(sum(t[0] for t in terms) - error, sum(t[1] for t in terms) + error)


def scalar(value):
    if not math.isfinite(value):
        raise Unsupported("nonfinite input")
    result = Q(value)
    return result, result


def packed(row, precision):
    return struct.pack("<" + ("f" if precision == "float" else "d") * len(row), *row)


def exact_distances(context, members, ar):
    """Prove an order-independent exact centroid and exact distance operations.

    If the proof fails, return unknown rather than call reference ties computed
    ties. Identical locations are handled separately for arbitrary centroids.
    """

    def exact(x):
        if x == 0:
            return x
        numerator = abs(x.numerator)
        while numerator % 2 == 0:
            numerator //= 2
        if (
            x.denominator & (x.denominator - 1)
            or numerator.bit_length() > ar.bits
            or abs(x) < ar.tiny
            or abs(x) > ar.maximum
        ):
            raise Unsupported("exact arithmetic proof unavailable")
        return x

    def unordered_sum(terms):
        denominator = max(x.denominator for x in terms)
        magnitude = sum(abs(x.numerator * (denominator // x.denominator)) for x in terms)
        if magnitude.bit_length() > ar.bits or Q(1, denominator) < ar.tiny:
            raise Unsupported("exact sum proof unavailable")
        return exact(sum(terms))

    try:
        weights = [exact(Q(w)) for w in context["feature_weights"]]
        points = [[Q(x) for x in context["points"][i]] for i in members]
        if context["representative"] == "medoid":
            locations = [[exact(x * w) for x, w in zip(row, weights)] for row in points]
            total = exact(Q(len(members)))
            numerators = [unordered_sum(list(col)) for col in zip(*locations)]
        elif context["representative"] == "weighted_mean" and not weights:
            locations = [[exact(Q(x)) for x in context["detection_idx"][i]] for i in members]
            total = exact(Q(len(members)))
            numerators = [unordered_sum(list(col)) for col in zip(*locations)]
        else:
            return None
        inv = exact(1 / total)
        centroid = [exact(x * inv) for x in numerators]
        distances = []
        for row in locations:
            distance = Q(0)
            for x, c in zip(row, centroid):
                delta = exact(x - c)
                distance = exact(distance + exact(delta * delta))
            distances.append(distance)
        return distances
    except Unsupported:
        return None


def validate_frame(context, precision, representatives, reduced=None):
    """Validate decoded rows. Returns auditable admissible rows and score bounds.

    Context contains actual typed inputs plus effective per-frame parameters.
    Custom scores are deliberately unresolved: caller device arithmetic
    requires a separately maintained reference model.
    """
    ar = Arithmetic(precision)
    labels, indices, points = (context[k] for k in ("labels", "detection_idx", "points"))
    policy = context["representative"]
    weights = context["feature_weights"]
    method = context["score_method"]
    if policy not in {"first", "peak", "medoid", "weighted_mean"}:
        raise Unsupported("unknown representative policy")
    if len(labels) != len(indices) or len(labels) != len(points):
        raise ValueError("input row alignment changed")
    features = len(points[0]) if points else len(weights)
    cols = len(indices[0]) if indices else 2
    if not 2 <= cols <= 10 or any(len(row) != cols for row in indices):
        raise ValueError("invalid index columns")
    if any(len(row) != features for row in points):
        raise ValueError("invalid feature rows")
    if any(type(lab) is not int or lab < -1 for lab in labels):
        raise ValueError("invalid labels")
    if any(type(x) is not int for row in indices for x in row):
        raise ValueError("indices must be integers")
    if policy in {"peak", "medoid"} and len(weights) != features:
        raise ValueError("missing feature weights")
    if weights and len(weights) != features:
        raise ValueError("feature weight shape changed")
    groups = {}
    for row, label in enumerate(labels):
        if label >= 0:
            groups.setdefault(label, []).append(row)
    if len(representatives) != len(groups) or (reduced is not None and len(reduced) != len(groups)):
        raise ValueError("occupied cluster count changed")
    receipts = []
    for out, (label, members) in enumerate(sorted(groups.items())):
        selected = [i for i in members if indices[i] == representatives[out]]
        if not selected:
            raise ValueError("representative is not a member of the ordered cluster")
        if policy == "peak":
            # Peak is deterministic; caller retains exact pair comparison and
            # existing analytic/tie GPU fixtures. Never relax peak indices.
            if reduced is not None and not any(
                packed(reduced[out], precision) == packed(points[i], precision) for i in selected
            ):
                raise ValueError("gathered features do not match selected member")
            receipts.append({"label": label, "exact_required": True})
            continue
        if policy == "first":
            admissible = [members[0]]
            means = None
            distances = None
        else:
            scaled = [
                [ar.mul(scalar(x), ar.rounded(Q(w), Q(w))) for x, w in zip(points[i], weights)]
                for i in members
            ]
            if policy == "medoid":
                inv = ar.reciprocal(ar.rounded(Q(len(members)), Q(len(members))))
                means = [ar.mul(ar.sum(list(col)), inv) for col in zip(*scaled)]
                locations = scaled
            else:
                if not weights:
                    alphas = [scalar(1) for _ in members]
                elif method == "weighted_sum":
                    alphas = []
                    for row in scaled:
                        score = scalar(0)
                        for term in row:
                            score = ar.add(score, term)
                        alphas.append(score)
                elif method == "weighted_l2":
                    alphas = []
                    for row in scaled:
                        score = scalar(0)
                        for term in row:
                            score = ar.add(score, ar.square(term))
                        alphas.append(ar.sqrt(score))
                elif method == "weighted_max":
                    alphas = [(max(x[0] for x in row), max(x[1] for x in row)) for row in scaled]
                else:
                    raise Unsupported("score reference unavailable: " + method)
                if any(a[0] < 0 for a in alphas):
                    raise Unsupported("row weight not provably nonnegative")
                inv = ar.reciprocal(ar.sum(alphas))

                def mean_columns(rows):
                    return [
                        ar.mul(
                            ar.sum(
                                [ar.mul(a, ar.rounded(Q(x), Q(x))) for a, x in zip(alphas, col)]
                            ),
                            inv,
                        )
                        for col in zip(*rows)
                    ]

                means = mean_columns([indices[i] for i in members])
                feature_means = mean_columns([points[i] for i in members])
                locations = [[ar.rounded(Q(x), Q(x)) for x in indices[i]] for i in members]
            distances = []
            for row in locations:
                distance = scalar(0)
                for x, cen in zip(row, means):
                    delta = ar.rounded(x[0] - cen[1], x[1] - cen[0])
                    distance = ar.add(distance, ar.square(delta))
                distances.append(distance)
            proven = exact_distances(context, members, ar)
            if proven is not None:
                distances = [(x, x) for x in proven]
            ceiling = min(x[1] for x in distances)
            admissible = [i for i, d in zip(members, distances) if d[0] <= ceiling]
            if proven is not None:
                admissible = admissible[:1]
            # Identical locations yield computed ties for every centroid. Such
            # ties can be resolved without pretending interval overlap is a tie.
            eligible = set(admissible)
            seen = set()
            admissible = []
            for i, location in zip(members, locations):
                key = tuple(location)
                if key not in seen and i in eligible:
                    admissible.append(i)
                seen.add(key)

        chosen = [i for i in selected if i in admissible]
        if not chosen:
            raise ValueError("representative outside input-derived admissible set")
        if reduced is not None:
            if len(reduced[out]) != features:
                raise ValueError("reduced feature shape changed")
            if policy == "weighted_mean":
                if any(
                    not math.isfinite(x) or not lo <= Q(x) <= hi
                    for x, (lo, hi) in zip(reduced[out], feature_means)
                ):
                    raise ValueError("weighted mean outside input-derived bounds")
            elif not any(
                packed(reduced[out], precision) == packed(points[i], precision) for i in chosen
            ):
                raise ValueError("gathered features do not match selected member")
        receipts.append(
            {
                "label": label,
                "admissible_rows": admissible,
                "selected_rows": chosen,
                "distance_bounds": (
                    None if distances is None else [[str(a), str(b)] for a, b in distances]
                ),
            }
        )
    return receipts


def compare_case(before, after, precision):
    """Joint validation; a missing reference is a blocker, never an exact fallback."""
    from equivalence import tensor_bytes, values

    contexts = before.get("semantic_inputs")
    if not contexts or contexts != after.get("semantic_inputs"):
        return {"reason": "missing or changed semantic inputs", "classification": "blocked"}
    if set(before["outputs"]) != {"rep_idx", "reduced_points"}:
        raise ValueError("semantic policy requires both DetectionReduction ports")
    receipts = []

    def decode(emission, dtype):
        if emission["dtype"] not in dtype:
            raise ValueError("semantic input/output precision changed")
        data = values(tensor_bytes(emission), emission["dtype"])
        shape = emission["shape"]
        if len(shape) == 1:
            return data
        if len(shape) != 2 or shape[1] <= 0:
            raise ValueError("semantic tensor rank changed")
        return [data[i : i + shape[1]] for i in range(0, len(data), shape[1])]

    real = {"float32" if precision == "float" else "float64"}
    try:
        for side in (before, after):
            outputs = side["outputs"]
            reps = outputs["rep_idx"]["emissions"]
            features = outputs["reduced_points"]["emissions"]
            if not outputs["rep_idx"]["wired"] or len(reps) != len(contexts):
                raise ValueError("representative emission coverage changed")
            if outputs["reduced_points"]["wired"] and len(features) != len(contexts):
                raise ValueError("feature emission coverage changed")
            for frame, witness in enumerate(contexts):
                context = dict(witness["parameters"])
                for port in ("labels", "detection_idx", "points"):
                    context[port] = decode(
                        witness["inputs"][port], real if port == "points" else {"int64"}
                    )
                reduced = (
                    decode(features[frame], real) if outputs["reduced_points"]["wired"] else None
                )
                if (
                    reduced is not None
                    and features[frame]["shape"][1] != witness["inputs"]["points"]["shape"][1]
                ):
                    raise ValueError("feature extent changed")
                if reps[frame]["shape"][1] != witness["inputs"]["detection_idx"]["shape"][1]:
                    raise ValueError("index extent changed")
                receipts.append(
                    validate_frame(context, precision, decode(reps[frame], {"int64"}), reduced)
                )
        return {"receipts": receipts}
    except Unsupported as error:
        return {"reason": str(error), "classification": "blocked"}
    except (ValueError, KeyError, IndexError, OverflowError) as error:
        return {"reason": str(error), "classification": "correctness_failure"}
