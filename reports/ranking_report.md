## Ranking performance (test set)

9,043 customers, 1,058 subscribers (base rate 11.7%). Decile 1 = highest-scored 10% of customers.

| Decile | Customers | Subscribers | Avg predicted | Actual rate | Lift | Cum. % customers | Cum. % subscribers |
|---|---|---|---|---|---|---|---|
| 1 | 905 | 462 | 50.4% | 51.0% | 4.4x | 10% | 44% |
| 2 | 904 | 202 | 19.0% | 22.3% | 1.9x | 20% | 63% |
| 3 | 904 | 93 | 10.9% | 10.3% | 0.9x | 30% | 72% |
| 4 | 905 | 66 | 8.4% | 7.3% | 0.6x | 40% | 78% |
| 5 | 904 | 69 | 6.9% | 7.6% | 0.7x | 50% | 84% |
| 6 | 904 | 43 | 5.8% | 4.8% | 0.4x | 60% | 88% |
| 7 | 905 | 41 | 4.9% | 4.5% | 0.4x | 70% | 92% |
| 8 | 904 | 34 | 4.1% | 3.8% | 0.3x | 80% | 95% |
| 9 | 904 | 31 | 3.4% | 3.4% | 0.3x | 90% | 98% |
| 10 | 904 | 17 | 2.5% | 1.9% | 0.2x | 100% | 100% |

- Top 10% of customers capture **44%** of subscribers (lift 4.4x).
- Top 20% capture **63%**.
- Calls per subscriber: 8.5 calling everyone vs 2.7 calling the top 20% (**3.1x cheaper**).

### Call economics (illustrative: $5 per call, $100 per subscription)

- Break-even probability = 5 / 100 = **5%**.
- Calling everyone: net **$60,585**.
- Calling only customers scored >= 5% (5,783 calls, 64% of the list, capturing 90% of subscribers): net **$66,285**.
