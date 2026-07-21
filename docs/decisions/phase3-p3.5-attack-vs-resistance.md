# P3.5 — Attack-vs-Resistance/Immunity Confusion Rate

**Issue:** #19
**Class:** extraction confuses a damage type the creature *attacks* with for a
`resists`/`immune_to` edge (the graph carries the attack as a defense).

## Measured rate

3 discrimination probes in the frozen query set (`eval/queries.json`), tagged
with `attack_vs_resistance_confusion`:

| id | value | verdict |
|---|---|---|
| `ent-ankheg-resist` | `True` | confirmed extraction confusion — ankheg's `resists ácido` edge is wrong; ácido is the ankheg's own acid *attack*, not a resistance (the known P1.4 error). |
| `ent-espectro-resist` | `False` | confirmed genuine — espectro's `resists`/`immune_to` edges are a real undead defense (correct contrast case). |
| `ent-deva-immune` | `False` | confirmed genuine — deva's `immune_to` edges are a real angelic immunity (correct contrast case). |

**1 confirmed confusion out of 3 discrimination probes → error rate ≈ 0.333.**

## Does this justify a prompt-hardening follow-up?

The measured rate **does not justify** a prompt-hardening follow-up in this
slice. This is a small, hand-designed probe set (n=3) built to discriminate
the known ANKHEG case against two verified-correct contrasts — not a random,
source-verified sample of the full graph. A single designed positive out of
three designed probes is expected by construction and says nothing about the
population error rate. The trigger for a follow-up would be a larger,
source-verified sample (drawn independently of the known ANKHEG case) showing
a materially higher rate.

## Scope note

No PDF corpus is available in this environment to verify or expand this
sample further at this time.
