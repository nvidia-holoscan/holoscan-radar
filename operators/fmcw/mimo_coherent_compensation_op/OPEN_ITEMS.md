# MIMOCoherentCompensationOp — open items

Real gaps only. Algorithm pins live in the operator README and
`OPERATOR_API.md` (`DDMCodeDescriptor` imposed layout). Aliased ÷M velocity
(no unfold) is the v1 product, not an open item.

## 1. Identity `tx_order` only (TDM)

**Class:** Correctness gap; cross-operator.

**Current behavior:** The TDM correction indexes the firing slot by TX axis
order (`m × PRI / M`). That is only correct when TX id equals firing slot
(default round-robin). `MIMODeinterleaveOp` accepts a uniform-repeating
`tx_order` permutation and orders its output TX axis by TX id. This operator
never receives `tx_order`, so a permuted schedule gets the **wrong phase,
silently**. Staggered / non-uniform TDM is a separate rejection (STATUS.md L12;
deinterleave `OPEN_ITEMS.md` B.1).

**Question:** How should the firing-slot map reach this operator — a
`tx_order` Param, a field on an existing descriptor, or a map emitted by
deinterleave?

**Acceptance:** A non-identity uniform `tx_order` applies the phase for each
TX's actual firing slot. A test must show a permutation that differs from
identity still aligns lanes after compensation. No latch. Do not fold staggered
TDM into this item.
