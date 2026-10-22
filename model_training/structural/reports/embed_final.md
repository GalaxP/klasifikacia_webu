# Language-agnostic HTML topic classifier — evaluation

- pages: 8291
- labels: Adult, Automotive, Computers, Entertainment, Finance, Food, Health, News, Shopping, Sports, Travel

**domain** — held-out websites (70/15/15 by registrable domain).  **cs2en / en2cs** — trained on all Czech pages, tested on all English ones, and the reverse: the language-independence test.  The two halves differ in vintage as well as language, so read those two as a worst case.  Per-label thresholds are tuned on validation rows only.

| config | split | n | micro F1 | macro F1 | exact | mean AUC | empty |
|---|---|---|---|---|---|---|---|
| `struct+text+embed` | domain | 1223 | 0.734 | 0.731 | 0.597 | 0.950 | 0.000 |
| `struct+text+embed` | cs2en | 2815 | 0.678 | 0.666 | 0.579 | 0.921 | 0.000 |
| `struct+text+embed` | en2cs | 5476 | 0.674 | 0.645 | 0.551 | 0.914 | 0.000 |
| `struct+embed` | domain | 1223 | 0.704 | 0.707 | 0.552 | 0.941 | 0.000 |
| `struct+embed` | cs2en | 2815 | 0.678 | 0.656 | 0.567 | 0.919 | 0.000 |
| `struct+embed` | en2cs | 5476 | 0.654 | 0.623 | 0.491 | 0.910 | 0.000 |
| `embed_only` | domain | 1223 | 0.681 | 0.678 | 0.512 | 0.927 | 0.000 |
| `embed_only` | cs2en | 2815 | 0.649 | 0.631 | 0.571 | 0.909 | 0.000 |
| `embed_only` | en2cs | 5476 | 0.641 | 0.611 | 0.444 | 0.907 | 0.000 |

## Per-class detail

### `struct+text+embed` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.844 | 0.753 | 0.796 |
| Entertainment | 216 | 0.691 | 0.796 | 0.740 |
| News | 208 | 0.671 | 0.764 | 0.715 |
| Computers | 194 | 0.744 | 0.825 | 0.782 |
| Health | 139 | 0.691 | 0.806 | 0.744 |
| Finance | 111 | 0.708 | 0.568 | 0.630 |
| Sports | 110 | 0.845 | 0.545 | 0.663 |
| Food | 101 | 0.755 | 0.733 | 0.744 |
| Travel | 96 | 0.573 | 0.656 | 0.612 |
| Automotive | 55 | 0.774 | 0.745 | 0.759 |
| Adult | 16 | 1.000 | 0.750 | 0.857 |

### `struct+text+embed` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.766 | 0.652 | 0.704 |
| Computers | 529 | 0.636 | 0.830 | 0.720 |
| Entertainment | 516 | 0.628 | 0.787 | 0.699 |
| Health | 299 | 0.678 | 0.706 | 0.692 |
| News | 207 | 0.561 | 0.488 | 0.522 |
| Sports | 203 | 0.721 | 0.714 | 0.718 |
| Finance | 192 | 0.591 | 0.625 | 0.608 |
| Travel | 183 | 0.627 | 0.568 | 0.596 |
| Food | 159 | 0.730 | 0.528 | 0.613 |
| Automotive | 89 | 0.575 | 0.517 | 0.544 |
| Adult | 55 | 0.925 | 0.891 | 0.907 |

### `struct+text+embed` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.815 | 0.753 | 0.783 |
| Computers | 1027 | 0.665 | 0.865 | 0.752 |
| News | 931 | 0.641 | 0.574 | 0.605 |
| Entertainment | 695 | 0.716 | 0.627 | 0.669 |
| Finance | 691 | 0.609 | 0.647 | 0.627 |
| Health | 514 | 0.755 | 0.539 | 0.629 |
| Travel | 499 | 0.529 | 0.717 | 0.609 |
| Food | 354 | 0.609 | 0.537 | 0.571 |
| Sports | 353 | 0.659 | 0.586 | 0.621 |
| Automotive | 215 | 0.529 | 0.595 | 0.560 |
| Adult | 72 | 0.851 | 0.556 | 0.672 |

### `struct+embed` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.807 | 0.740 | 0.772 |
| Entertainment | 216 | 0.663 | 0.801 | 0.725 |
| News | 208 | 0.653 | 0.707 | 0.679 |
| Computers | 194 | 0.679 | 0.830 | 0.747 |
| Health | 139 | 0.654 | 0.763 | 0.704 |
| Finance | 111 | 0.631 | 0.586 | 0.607 |
| Sports | 110 | 0.795 | 0.527 | 0.634 |
| Food | 101 | 0.679 | 0.752 | 0.714 |
| Travel | 96 | 0.480 | 0.635 | 0.547 |
| Automotive | 55 | 0.732 | 0.745 | 0.739 |
| Adult | 16 | 0.933 | 0.875 | 0.903 |

### `struct+embed` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.731 | 0.722 | 0.726 |
| Computers | 529 | 0.651 | 0.815 | 0.724 |
| Entertainment | 516 | 0.692 | 0.731 | 0.711 |
| Health | 299 | 0.674 | 0.699 | 0.686 |
| News | 207 | 0.545 | 0.498 | 0.520 |
| Sports | 203 | 0.690 | 0.724 | 0.707 |
| Finance | 192 | 0.532 | 0.646 | 0.584 |
| Travel | 183 | 0.538 | 0.617 | 0.575 |
| Food | 159 | 0.650 | 0.503 | 0.567 |
| Automotive | 89 | 0.573 | 0.483 | 0.524 |
| Adult | 55 | 0.891 | 0.891 | 0.891 |

### `struct+embed` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.779 | 0.748 | 0.763 |
| Computers | 1027 | 0.674 | 0.841 | 0.749 |
| News | 931 | 0.584 | 0.640 | 0.611 |
| Entertainment | 695 | 0.692 | 0.665 | 0.678 |
| Finance | 691 | 0.575 | 0.622 | 0.598 |
| Health | 514 | 0.710 | 0.591 | 0.645 |
| Travel | 499 | 0.582 | 0.641 | 0.610 |
| Food | 354 | 0.514 | 0.562 | 0.537 |
| Sports | 353 | 0.519 | 0.635 | 0.571 |
| Automotive | 215 | 0.303 | 0.707 | 0.425 |
| Adult | 72 | 1.000 | 0.500 | 0.667 |

### `embed_only` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.704 | 0.736 | 0.720 |
| Entertainment | 216 | 0.661 | 0.787 | 0.719 |
| News | 208 | 0.692 | 0.668 | 0.680 |
| Computers | 194 | 0.664 | 0.804 | 0.727 |
| Health | 139 | 0.640 | 0.755 | 0.693 |
| Finance | 111 | 0.654 | 0.459 | 0.540 |
| Sports | 110 | 0.639 | 0.627 | 0.633 |
| Food | 101 | 0.699 | 0.713 | 0.706 |
| Travel | 96 | 0.483 | 0.604 | 0.537 |
| Automotive | 55 | 0.627 | 0.582 | 0.604 |
| Adult | 16 | 0.933 | 0.875 | 0.903 |

### `embed_only` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.734 | 0.639 | 0.683 |
| Computers | 529 | 0.618 | 0.824 | 0.706 |
| Entertainment | 516 | 0.767 | 0.630 | 0.691 |
| Health | 299 | 0.750 | 0.552 | 0.636 |
| News | 207 | 0.537 | 0.386 | 0.449 |
| Sports | 203 | 0.597 | 0.759 | 0.668 |
| Finance | 192 | 0.574 | 0.604 | 0.589 |
| Travel | 183 | 0.486 | 0.656 | 0.558 |
| Food | 159 | 0.608 | 0.547 | 0.576 |
| Automotive | 89 | 0.489 | 0.506 | 0.497 |
| Adult | 55 | 0.875 | 0.891 | 0.883 |

### `embed_only` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.731 | 0.734 | 0.732 |
| Computers | 1027 | 0.656 | 0.838 | 0.736 |
| News | 931 | 0.557 | 0.668 | 0.607 |
| Entertainment | 695 | 0.633 | 0.702 | 0.666 |
| Finance | 691 | 0.581 | 0.644 | 0.611 |
| Health | 514 | 0.592 | 0.658 | 0.623 |
| Travel | 499 | 0.479 | 0.711 | 0.573 |
| Food | 354 | 0.501 | 0.593 | 0.543 |
| Sports | 353 | 0.603 | 0.629 | 0.616 |
| Automotive | 215 | 0.272 | 0.716 | 0.394 |
| Adult | 72 | 0.818 | 0.500 | 0.621 |
