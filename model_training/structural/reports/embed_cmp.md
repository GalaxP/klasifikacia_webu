# Language-agnostic HTML topic classifier — evaluation

- pages: 8291
- labels: Adult, Automotive, Computers, Entertainment, Finance, Food, Health, News, Shopping, Sports, Travel

**domain** — held-out websites (70/15/15 by registrable domain).  **cs2en / en2cs** — trained on all Czech pages, tested on all English ones, and the reverse: the language-independence test.  The two halves differ in vintage as well as language, so read those two as a worst case.  Per-label thresholds are tuned on validation rows only.

| config | split | n | micro F1 | macro F1 | exact | mean AUC | empty |
|---|---|---|---|---|---|---|---|
| `text+embed` | domain | 1223 | 0.762 | 0.754 | 0.612 | 0.960 | 0.000 |
| `text+embed` | cs2en | 2815 | 0.529 | 0.497 | 0.473 | 0.904 | 0.000 |
| `text+embed` | en2cs | 5476 | 0.633 | 0.591 | 0.547 | 0.908 | 0.000 |
| `text_only` | domain | 1223 | 0.757 | 0.744 | 0.615 | 0.956 | 0.000 |
| `text_only` | cs2en | 2815 | 0.506 | 0.467 | 0.446 | 0.875 | 0.000 |
| `text_only` | en2cs | 5476 | 0.591 | 0.551 | 0.463 | 0.896 | 0.000 |
| `struct+embed` | domain | 1223 | 0.529 | 0.514 | 0.358 | 0.844 | 0.000 |
| `struct+embed` | cs2en | 2815 | 0.369 | 0.260 | 0.312 | 0.743 | 0.000 |
| `struct+embed` | en2cs | 5476 | 0.329 | 0.313 | 0.208 | 0.723 | 0.000 |
| `all_struct` | domain | 1223 | 0.495 | 0.465 | 0.348 | 0.807 | 0.000 |
| `all_struct` | cs2en | 2815 | 0.323 | 0.213 | 0.272 | 0.682 | 0.000 |
| `all_struct` | en2cs | 5476 | 0.281 | 0.295 | 0.073 | 0.680 | 0.000 |

## Per-class detail

### `text+embed` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.854 | 0.795 | 0.824 |
| Entertainment | 216 | 0.744 | 0.847 | 0.792 |
| News | 208 | 0.676 | 0.894 | 0.770 |
| Computers | 194 | 0.761 | 0.871 | 0.812 |
| Health | 139 | 0.621 | 0.885 | 0.730 |
| Finance | 111 | 0.664 | 0.766 | 0.711 |
| Sports | 110 | 0.750 | 0.573 | 0.649 |
| Food | 101 | 0.723 | 0.723 | 0.723 |
| Travel | 96 | 0.628 | 0.740 | 0.679 |
| Automotive | 55 | 0.810 | 0.618 | 0.701 |
| Adult | 16 | 1.000 | 0.812 | 0.897 |

### `text+embed` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.749 | 0.371 | 0.496 |
| Computers | 529 | 0.598 | 0.851 | 0.703 |
| Entertainment | 516 | 0.357 | 0.884 | 0.508 |
| Health | 299 | 0.774 | 0.411 | 0.537 |
| News | 207 | 0.467 | 0.169 | 0.248 |
| Sports | 203 | 0.833 | 0.296 | 0.436 |
| Finance | 192 | 0.774 | 0.339 | 0.471 |
| Travel | 183 | 0.833 | 0.301 | 0.442 |
| Food | 159 | 0.939 | 0.289 | 0.442 |
| Automotive | 89 | 0.590 | 0.258 | 0.359 |
| Adult | 55 | 1.000 | 0.709 | 0.830 |

### `text+embed` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.842 | 0.750 | 0.793 |
| Computers | 1027 | 0.667 | 0.869 | 0.755 |
| News | 931 | 0.775 | 0.207 | 0.327 |
| Entertainment | 695 | 0.602 | 0.692 | 0.644 |
| Finance | 691 | 0.601 | 0.550 | 0.574 |
| Health | 514 | 0.452 | 0.642 | 0.531 |
| Travel | 499 | 0.707 | 0.625 | 0.664 |
| Food | 354 | 0.519 | 0.534 | 0.526 |
| Sports | 353 | 0.679 | 0.473 | 0.558 |
| Automotive | 215 | 0.937 | 0.274 | 0.424 |
| Adult | 72 | 0.770 | 0.653 | 0.707 |

### `text_only` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.837 | 0.823 | 0.830 |
| Entertainment | 216 | 0.720 | 0.847 | 0.779 |
| News | 208 | 0.698 | 0.832 | 0.759 |
| Computers | 194 | 0.776 | 0.856 | 0.814 |
| Health | 139 | 0.611 | 0.871 | 0.718 |
| Finance | 111 | 0.683 | 0.739 | 0.710 |
| Sports | 110 | 0.787 | 0.536 | 0.638 |
| Food | 101 | 0.753 | 0.663 | 0.705 |
| Travel | 96 | 0.634 | 0.740 | 0.683 |
| Automotive | 55 | 0.805 | 0.600 | 0.688 |
| Adult | 16 | 0.929 | 0.812 | 0.867 |

### `text_only` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.728 | 0.366 | 0.487 |
| Computers | 529 | 0.589 | 0.839 | 0.692 |
| Entertainment | 516 | 0.338 | 0.876 | 0.488 |
| Health | 299 | 0.732 | 0.338 | 0.462 |
| News | 207 | 0.446 | 0.159 | 0.235 |
| Sports | 203 | 0.814 | 0.236 | 0.366 |
| Finance | 192 | 0.773 | 0.302 | 0.434 |
| Travel | 183 | 0.875 | 0.268 | 0.410 |
| Food | 159 | 0.922 | 0.296 | 0.448 |
| Automotive | 89 | 0.500 | 0.236 | 0.321 |
| Adult | 55 | 1.000 | 0.655 | 0.791 |

### `text_only` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.836 | 0.701 | 0.763 |
| Computers | 1027 | 0.526 | 0.911 | 0.667 |
| News | 931 | 0.697 | 0.183 | 0.289 |
| Entertainment | 695 | 0.599 | 0.658 | 0.627 |
| Finance | 691 | 0.517 | 0.605 | 0.558 |
| Health | 514 | 0.387 | 0.654 | 0.486 |
| Travel | 499 | 0.682 | 0.579 | 0.626 |
| Food | 354 | 0.496 | 0.497 | 0.496 |
| Sports | 353 | 0.557 | 0.487 | 0.520 |
| Automotive | 215 | 0.518 | 0.400 | 0.451 |
| Adult | 72 | 0.714 | 0.486 | 0.579 |

### `struct+embed` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.740 | 0.681 | 0.709 |
| Entertainment | 216 | 0.523 | 0.685 | 0.593 |
| News | 208 | 0.587 | 0.630 | 0.608 |
| Computers | 194 | 0.544 | 0.670 | 0.600 |
| Health | 139 | 0.278 | 0.647 | 0.389 |
| Finance | 111 | 0.312 | 0.405 | 0.353 |
| Sports | 110 | 0.462 | 0.491 | 0.476 |
| Food | 101 | 0.350 | 0.416 | 0.380 |
| Travel | 96 | 0.327 | 0.385 | 0.354 |
| Automotive | 55 | 0.438 | 0.382 | 0.408 |
| Adult | 16 | 0.917 | 0.688 | 0.786 |

### `struct+embed` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.623 | 0.551 | 0.585 |
| Computers | 529 | 0.390 | 0.571 | 0.463 |
| Entertainment | 516 | 0.439 | 0.339 | 0.383 |
| Health | 299 | 0.229 | 0.241 | 0.235 |
| News | 207 | 0.206 | 0.396 | 0.271 |
| Sports | 203 | 0.390 | 0.202 | 0.266 |
| Finance | 192 | 0.188 | 0.224 | 0.204 |
| Travel | 183 | 0.178 | 0.208 | 0.191 |
| Food | 159 | 0.294 | 0.063 | 0.104 |
| Automotive | 89 | 0.150 | 0.067 | 0.093 |
| Adult | 55 | 0.094 | 0.055 | 0.069 |

### `struct+embed` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.718 | 0.575 | 0.638 |
| Computers | 1027 | 0.508 | 0.516 | 0.512 |
| News | 931 | 0.548 | 0.184 | 0.275 |
| Entertainment | 695 | 0.349 | 0.459 | 0.397 |
| Finance | 691 | 0.266 | 0.136 | 0.180 |
| Health | 514 | 0.179 | 0.163 | 0.171 |
| Travel | 499 | 0.231 | 0.122 | 0.160 |
| Food | 354 | 0.119 | 0.415 | 0.185 |
| Sports | 353 | 0.351 | 0.246 | 0.290 |
| Automotive | 215 | 0.065 | 0.633 | 0.117 |
| Adult | 72 | 0.778 | 0.389 | 0.519 |

### `all_struct` / domain

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 288 | 0.724 | 0.646 | 0.683 |
| Entertainment | 216 | 0.515 | 0.630 | 0.567 |
| News | 208 | 0.554 | 0.615 | 0.583 |
| Computers | 194 | 0.467 | 0.629 | 0.536 |
| Health | 139 | 0.333 | 0.237 | 0.277 |
| Finance | 111 | 0.270 | 0.387 | 0.319 |
| Sports | 110 | 0.356 | 0.564 | 0.437 |
| Food | 101 | 0.414 | 0.238 | 0.302 |
| Travel | 96 | 0.300 | 0.219 | 0.253 |
| Automotive | 55 | 0.329 | 0.418 | 0.368 |
| Adult | 16 | 0.917 | 0.688 | 0.786 |

### `all_struct` / cs2en

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 715 | 0.633 | 0.520 | 0.571 |
| Computers | 529 | 0.331 | 0.524 | 0.406 |
| Entertainment | 516 | 0.377 | 0.260 | 0.308 |
| Health | 299 | 0.160 | 0.171 | 0.165 |
| News | 207 | 0.174 | 0.377 | 0.238 |
| Sports | 203 | 0.298 | 0.153 | 0.202 |
| Finance | 192 | 0.148 | 0.198 | 0.170 |
| Travel | 183 | 0.160 | 0.115 | 0.134 |
| Food | 159 | 0.091 | 0.019 | 0.031 |
| Automotive | 89 | 0.200 | 0.034 | 0.058 |
| Adult | 55 | 0.083 | 0.055 | 0.066 |

### `all_struct` / en2cs

| label | n | P | R | F1 |
|---|---|---|---|---|
| Shopping | 1324 | 0.766 | 0.530 | 0.627 |
| Computers | 1027 | 0.547 | 0.371 | 0.442 |
| News | 931 | 0.422 | 0.432 | 0.427 |
| Entertainment | 695 | 0.318 | 0.449 | 0.372 |
| Finance | 691 | 0.223 | 0.075 | 0.113 |
| Health | 514 | 0.117 | 0.399 | 0.180 |
| Travel | 499 | 0.245 | 0.106 | 0.148 |
| Food | 354 | 0.092 | 0.432 | 0.152 |
| Sports | 353 | 0.246 | 0.232 | 0.239 |
| Automotive | 215 | 0.045 | 0.726 | 0.084 |
| Adult | 72 | 0.750 | 0.333 | 0.462 |
