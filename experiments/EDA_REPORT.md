# Business Entity Resolution — Exploratory Data Analysis & Evaluation Foundation Report

**Project:** Business Entity Resolution Competition  
**Phase:** Phase 1 — Data Loading, Profiling, GT Analysis, Validation Split & Metric Infrastructure  
**Date:** September 2026  
**Status:** Completed  

---

## 1. Dataset Overview
The Business Entity Resolution challenge requires matching real-world business entities across three independent, noisy data sources:
- **Source 1 (`S1`)**: Deduplicated reference source. Each entity in Source 1 requires finding all matching records in Source 2 and Source 3.
- **Source 2 (`S2`)**: Second business data source.
- **Source 3 (`S3`)**: Third business data source.
- **Ground Truth (`train_ground_truth.tsv`)**: Contains the true mapping from each Source 1 training entity ID to a comma-separated list of matching Source 2 and Source 3 entity IDs.

---

## 2. Dataset Sizes & Records Count
All datasets were loaded using `pd.read_csv(filepath, sep="\t")`.

| Dataset File | File Type | Total Rows | Total Columns | Unique `entity_id` Count | Duplicate `entity_id` Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dataset/train/train_source1.tsv` | Training Source 1 | 2,206,821 | 4 | 2,206,821 | 0 |
| `dataset/train/train_source2.tsv` | Training Source 2 | 5,034,616 | 4 | 5,034,616 | 0 |
| `dataset/train/train_source3.tsv` | Training Source 3 | 5,285,603 | 4 | 5,285,603 | 0 |
| `dataset/train/train_ground_truth.tsv` | Training Ground Truth | 2,206,821 | 2 | 2,206,821 | 0 |
| `dataset/test/test_source1.tsv` | Test Source 1 | 1,732,544 | 4 | 1,732,544 | 0 |
| `dataset/test/test_source2.tsv` | Test Source 2 | 4,887,273 | 4 | 4,887,273 | 0 |
| `dataset/test/test_source3.tsv` | Test Source 3 | 5,082,316 | 4 | 5,082,316 | 0 |

- **Total Training Records Across Sources**: 12,527,040 rows
- **Total Test Records Across Sources**: 11,702,133 rows
- **Grand Total Business Records**: 24,229,173 rows

---

## 3. Schema & Data Types

### Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `entity_id` | `object` (`str`) | Unique identifier prefixed with source prefix (`S1-`, `S2-`, `S3-`) |
| `business_name` | `object` (`str`) | Name of the business entity |
| `business_address` | `object` (`str`) | Address of the business entity |
| `country` | `object` (`str`) | Open-set country string label (`US`, `India`, `France`) |

### Ground Truth File (`train_ground_truth.tsv`)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `source1_entity_id` | `object` (`str`) | Source 1 reference entity identifier |
| `matched_entity_ids` | `object` (`str`) | Comma-separated list of matching S2/S3 entity IDs (or empty/NaN for singletons) |

---

## 4. Missing Values & String Length Profiling

### Missing Values Summary
| Dataset | `entity_id` Missing | `business_name` Missing | `business_address` Missing | `country` Missing |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2` | 0 (0.00%) | 2 (<0.01%) | 168,967 (3.36%) | 0 (0.00%) |
| `train_source3` | 0 (0.00%) | 13 (<0.01%) | 175,916 (3.33%) | 0 (0.00%) |
| `train_ground_truth` | 0 (0.00%) | N/A | N/A | N/A (123,247 NaN matched_entity_ids = singletons) |
| `test_source1` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2` | 0 (0.00%) | 46 (<0.01%) | 129,408 (2.65%) | 0 (0.00%) |
| `test_source3` | 0 (0.00%) | 59 (<0.01%) | 136,098 (2.68%) | 0 (0.00%) |

### String Length Profiling (`business_name` & `business_address`)
| Dataset | Column | Min Length | Median Length | Mean Length | Max Length | Missing/Empty % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1` | `business_name` | 2 | 26.0 | 27.84 | 225 | 0.00% |
| `train_source1` | `business_address` | 4 | 55.0 | 58.42 | 344 | 0.00% |
| `train_source2` | `business_name` | 2 | 26.0 | 27.85 | 231 | <0.01% |
| `train_source2` | `business_address` | 3 | 55.0 | 58.43 | 350 | 3.36% |
| `train_source3` | `business_name` | 2 | 26.0 | 27.85 | 230 | <0.01% |
| `train_source3` | `business_address` | 3 | 55.0 | 58.43 | 348 | 3.33% |
| `test_source1` | `business_name` | 2 | 25.0 | 26.91 | 215 | 0.00% |
| `test_source1` | `business_address` | 4 | 52.0 | 55.61 | 338 | 0.00% |
| `test_source2` | `business_name` | 2 | 25.0 | 26.92 | 220 | <0.01% |
| `test_source2` | `business_address` | 3 | 52.0 | 55.62 | 342 | 2.65% |
| `test_source3` | `business_name` | 2 | 25.0 | 26.92 | 218 | <0.01% |
| `test_source3` | `business_address` | 3 | 52.0 | 55.62 | 340 | 2.68% |

---

## 5. Duplicate Analysis
- `0` duplicate entity IDs in all source files.
- `0` duplicate `source1_entity_id` rows in ground truth.
- Reference integrity is intact.

---

## 6. Country Distributions & Open-Set Requirement

### Country Breakdown Across All Sources
| Source | US Count (%) | India Count (%) | France Count (%) | Total Rows |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1` | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2,206,821 |
| `train_source2` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 5,034,616 |
| `train_source3` | 3,170,056 (59.97%) | 2,115,547 (40.03%) | 0 (0.00%) | 5,285,603 |
| **Total Train** | **7,510,506 (59.95%)** | **5,016,534 (40.05%)** | **0 (0.00%)** | **12,527,040** |
| `test_source1` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 1,732,544 |
| `test_source2` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 4,887,273 |
| `test_source3` | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.39%) | 5,082,316 |
| **Total Test** | **4,480,137 (38.28%)** | **5,527,551 (47.24%)** | **1,694,445 (14.48%)** | **11,702,133** |

> [!IMPORTANT]
> **Open-Set Country Finding:** `France` is present ONLY in the test set (14.98% of test S1 records). Country must be treated as an open-set string label. Pipelines must never hardcode allowed countries to `{US, India}` or filter out `France`.

---

## 7. Ground-Truth Match Cardinality & S2/S3 Distribution

### Match-Count Buckets (Source 1 Entities)
| Bucket | Match Count Range | Number of S1 Entities | Percentage of S1 Entities |
| :--- | :--- | :--- | :--- |
| **0 Matches** (Singletons) | 0 | 123,247 | 5.58% |
| **1 Match** | 1 | 119,157 | 5.40% |
| **2+ Matches** | >= 2 | 1,964,417 | 89.02% |
| **Total** | | **2,206,821** | **100.00%** |

### Summary Match Statistics
- **Average Matches per S1 Entity**: 3.4613
- **Median Matches per S1 Entity**: 3.0
- **Maximum Matches per S1 Entity**: 11
- **Total Ground-Truth Positive Pairs**: 7,638,365

### Source Breakdown of Matches
- **S2-Only Matches**: 143,029 S1 entities (6.48%)
- **S3-Only Matches**: 164,498 S1 entities (7.45%)
- **S2 + S3 Combined Matches**: 1,776,047 S1 entities (80.48%)
- **Total Referenced S2 Entities**: 3,693,619
- **Total Referenced S3 Entities**: 3,944,746

### Ground-Truth Consistency Verification
- **Invalid S2 Entity References**: 0 (100% exist in `train_source2.tsv`)
- **Invalid S3 Entity References**: 0 (100% exist in `train_source3.tsv`)
- **Conclusion**: Ground truth references have zero dangling pointer errors.

---

## 8. Positive-Pair Noise Analysis
Analysis of 100,000 sampled true positive pairs (`S1-S2` and `S1-S3` matches):

| Variation Metric | Business Name Exact Match % | Business Address Exact Match % |
| :--- | :--- | :--- |
| **Raw Exact Equality** | 4.61% | 2.24% |
| **Case-Insensitive Exact Equality** | 10.70% | 7.32% |
| **Punctuation-Insensitive Equality** | 14.58% | 8.08% |

> [!NOTE]
> **Key Insight**: Over **85%** of matched true entity pairs differ beyond casing and punctuation!

### Identified Name Noise Patterns
1. **Transliteration into Non-Latin Scripts**: Names written in regional scripts (e.g., Telugu `Baba టెక్నాలజీస్`, Devanagari `किसान आवास`).
2. **Legal Suffix Variations**: Inconsistent legal abbreviations (`Pvt Ltd` vs `Private Limited` vs `PVT LTD.`, `Inc` vs `P.C.`, `LLP` vs omitted).
3. **Typographical & OCR Errors**: Digit substitutions (`C0nstruction` with zero `0`), letter transpositions (`Sthoen` for `Stone`, `Aelxandria` for `Alexandria`, `Ahute` for `Haute`).
4. **Prefix & Honorific Noise**: Added titles or symbols (`*** UROLOGY`, `Sri ENCORE CABS`, `Smt ROYAL CONSULTANCY`).
5. **Token Reordering**: Words swapped (`The Dent Cafe Inc.` vs `The Cafe Dent Inc.`, `LLC Eye Clinic` vs `Eye Clinic LLC`).
6. **Abbreviations & Parenthetical Noise**: `Evans Pediatrics` vs `Evans (Pediatrics)`.

### Identified Address Noise Patterns
1. **Transliteration & Regional Script**: State/city names in regional script (`उत्तर प्रदेश`, `महाराष्ट्र`, `ಕರ್ನಾಟಕ`).
2. **State & Street Standardization**: Full state name vs 2-letter abbreviation (`Indiana` vs `IN`, `Minnesota` vs `MN`, `Road` vs `Rd`, `Drive` vs `Dr`, `Avenue` vs `Ave`).
3. **Component Reordering**: Address tokens permuted (`BEKELEY SPRINGS CDP, WV, 225 WOLF TRAIL` vs `225 Wolf Trail, Berkeley Springs, WV`).
4. **Leading Punctuation & Noise Symbols**: Added hash symbols or double hashes (`#4120 Highway 77`, `##2337 Pendragon Rd`).
5. **Missing Unit/PIN/City Details**: Partial addresses lacking ZIP codes or explicit city names.

---

## 9. Concrete Matched Examples Table (Part D)

The following 20 concrete matched pairs selected from training data demonstrate key noise patterns:

| S1 Entity ID | Matched Entity ID | S1 Business Name | Matched Business Name | S1 Business Address | Matched Business Address | Country | Key Noise Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S1-341462203` | `S2-219553948` | Baba Technologies Private Limited | Baba టెక్నాలజీస్ ప్రైవేట్ లిమిటెడ్ | 37-90 Squaradeal, Defence Colony Sainikapuri, Secunderabad, Hyderabad, Telangana | NO 37-90 SQUARADEAL, DEFENCE COLONY SAINIKAPURI, SECUNDERABAD, Andhra Pradesh | India | Non-Latin script transliteration (Telugu), case & state variation |
| `S1-783955316` | `S3-656241763` | Kisan Avas Ltd | Kisan Avas Ltd. | 6/161 S 2, Rajender Nagar, Ghaziabad, Uttar Pradesh | उत्तर प्रदेश, 7/161 S 2, Ghaziabad | India | Devanagari script transliteration, punctuation, street number typo |
| `S1-313221336` | `S3-872559960` | Studio 74 Book Store | Studio 74 BOOK Store | 2628 Mariposa Drive, Terre Haute, IN | Indiana, 02628 Mariposa Dr, Terre Ahute | US | Casing, state name expansion, street abbreviation, city typo |
| `S1-516394603` | `S2-642842175` | Bay Creative Stone LLC | Bay Creative Sthoen LLC | 4018 Annapolis Road, Unit APARTMENT B, Halethorpe, MD | 4018- Annapolis Rd, HALETHORPE, MD | US | Typo (`Sthoen` for `Stone`), address unit truncation |
| `S1-887045319` | `S2-612158384` | Construction Vr Certifications Pvt Ltd | C0nstruction Vr Certhificatidons Pvt Ltd | H.No1-8-67/P Apiic Kamalanagar, Kushaiguda, Hyderabad, Rangareddy, Telangana | #1-8-67/P APIIC KAMALANAGAR, KUSHAIGUDA, RANGAREDDY, Telangana | India | OCR digit error (`0` for `o`), typo (`Certhificatidons`), hash symbol |
| `S1-21082318` | `S2-234787766` | Eye Clinic LLC | LLC Eye Clinic | 2475 Pecos Road, Unit 2098, Chandler, AZ | 2475 Pecos Road, ANDERSEN SPRINGS, AZ | US | Legal suffix word transposition, neighborhood landmark vs city |
| `S1-214543067` | `S2-416909923` | Nieto Algonquin LLC | Nieto África LLC | 225 Wolf Trail, Berkeley Springs, WV | BEKELEY SPRINGS CDP, WV, 225 WOLF TRAIL | US | Accent characters, address component reordering |
| `S1-180257062` | `S3-618488662` | Alexandria Hunger Rapid League | Alexandria  Hunger Rapid League | 1821 Aga Drive, Alexandria, MN | 1821. Aga Drive, Aelxandria, Minnesota | US | Extra whitespace, city typo (`Aelxandria`), state expansion |
| `S1-151440370` | `S2-918927342` | Shlok Electronics Pvt Ltd | SHLOK ELECTRONICS | S. No. 192/5, Office 03, Arambh, Nr. Datta Mandir, Pune, Maharashtra | S. NO. 192/5, OFFICE 03, ARAMBH, NR. DATTA MANDIR, PUNE, Maharashtra | India | Omitted legal suffix (`Pvt Ltd`), casing |
| `S1-241438014` | `S3-828994416` | Evans Pediatrics | Evans  (Pediatrics) | 1600 2nd West Hill Drive, Great Falls, MT | 1600 2nd West Hill Dr, Great Falls, Montana | US | Parenthetical formatting, street abbreviation, state expansion |
| `S1-61084401` | `S3-905700617` | Urology Horizon Health LLC | *** UROLOGY HORIZON HEALTH LLC | Sunset, AR, 4120 Highway 77 | #4120 Highway 77, Marion, Arkansas | US | Prefix noise (`***`), leading hash (`#`), city vs neighborhood |
| `S1-957558089` | `S3-583741602` | The Dent Cafe Inc. | The Cafe Dent Inc. | 165 Clearview Road, Eden, NC | 279 Clearview Road, Eden, North Carolina | US | Token transposition (`Dent Cafe` vs `Cafe Dent`), street number variation |
| `S1-528651403` | `S3-7289261` | Aloysia Damico Generation Inc | Aloysia Generation Damico Inc | 2337 Pendragon Road, Kingsport, TN | ##2337 Pendragon Rd, Kingsport, Tennessee | US | Token reordering, double hash symbol (`##`), street abbreviation |
| `S1-815044399` | `S3-294159911` | Henderson Federal Sciences Inc | Henderson Féderal | 1522 E 7th Road, Lostant, IL | 1522 East 7nd Rd, Lostant, Illinois | US | Truncated business name, accented character, ordinal typo (`7nd`) |
| `S1-715861166` | `S2-705665414` | FF Realcon LLP | FF REALCON LLP | C 06 Apmc Market I Phase Ii Sector 19, Turbhe Vashi, Thane, Maharashtra | महाराष्ट्र, C ##06 APMC MARKET I PHASE II SECTOR 19, TURBHE VASHI, THANE | India | Regional script state prefix, double hash symbol, casing |
| `S1-322698341` | `S3-430209411` | My Blue Exports Pvt Ltd | My Blue Pvt Pvt Ltd Services | At Post Tanahali Tal Indi, Bijapur, Karnataka | Bijapur, ಕರ್ನಾಟಕ, Bijapur, At Post Tanahali Tal Indi | India | Repeated legal suffix tokens, Kannada script state label, token reordering |
| `S1-204545711` | `S2-347900826` | Chiropractic Care Associates of Pittsfield | CHIROPRACTIC CARE ASSOCIATES OF PITTSFIELD LP | 50 Bartlett Avenue, Unit 2, Pittsfield, MA | 50. BARTLETT AVE, PITTSFIELD, MA | US | Added legal suffix (`LP`), street abbreviation, dot formatting |
| `S1-305970695` | `S3-582801494` | Straight Edge Book Store P.C. | Straight Edge Book Store | 3035 River Birch Drive, Unit Unit A, City Of Brookfield, WI | 3035 River Birch Drive, Unit Unit A, Brookfield, Wisconsin | US | Legal suffix omission (`P.C.`), city prefix omission (`City Of`) |
| `S1-742019967` | `S2-48286361` | Encore Cabs Pvt Ltd | Sri ENCORE CABS PVT LTD. | S-1 Unit, Second Floor, Anr Centre No. 137, Road No.1, Banjara Hills, Hyderabad, Telangana | S-#1 UNIT, SECOND FLOOR, ANR CENTRE NO. 137, ROAD NO.1, BANJARA HILLS, Telangana | India | Prefix honorific (`Sri`), internal hash symbol |
| `S1-433108657` | `S2-677618114` | Royal Consultancy Private Limited | Smt ROYAL CONSULTANCY  PRIVATE LIMITED | Mumbai City, C/O City Light Cinema, Ashford Chambers, Lady Jamshedji Road, Mahim, Mumbai, Maharashtra | C/O CITY LIGHT CINEMA, ASHFORD CHAMBERS, LAY JAMSHEDJI ROAD, MAHIM, MUMBAI, Maharashtra | India | Prefix honorific (`Smt`), typo (`LAY` for `LADY`) |

---

## 10. Entity-Level Validation Split Design (Part E)

To prevent data leakage during candidate generation and matching model experiments, the validation split is constructed strictly at the **Source 1 entity level**.

- **Split Ratio**: 80% Training / 20% Validation
- **Fixed Random Seed**: `42`
- **Stratification**: Jointly stratified across `gt_bucket` (`0`, `1`, `2+` matches) and `country` (`US`, `India`).

### Validation Split Statistics Comparison
| Metric | Full Training Set | Train Split (80%) | Validation Split (20%) |
| :--- | :--- | :--- | :--- |
| **Total S1 Entities** | 2,206,821 | 1,765,456 (80.00%) | 441,365 (20.00%) |
| **0-Match Entities** | 123,247 (5.58%) | 98,598 (5.58%) | 24,649 (5.58%) |
| **1-Match Entities** | 119,157 (5.40%) | 95,326 (5.40%) | 23,831 (5.40%) |
| **2+ Match Entities** | 1,964,417 (89.02%) | 1,571,532 (89.02%) | 392,885 (89.02%) |
| **US S1 Count** | 1,323,633 (59.98%) | 1,058,906 (59.98%) | 264,727 (59.98%) |
| **India S1 Count** | 883,188 (40.02%) | 706,550 (40.02%) | 176,638 (40.02%) |
| **Total GT Matches** | 7,638,365 | 6,110,658 (80.00%) | 1,527,707 (20.00%) |

---

## 11. F0.5 Metric & Country-Level Evaluation Infrastructure (Part G)

### Competition Metric Definition
The competition evaluates entity resolution using the precision-heavy $F_{0.5}$ score macro-averaged per Source 1 entity:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Edge-Case Specification & Singleton Credit Behavior
- **True Singleton (`True = ∅`, `Predicted = ∅`)**: Receives full credit score of $1.0$ ($\text{Precision} = 1.0, \text{Recall} = 1.0$).
- **False Merge on Singleton (`True = ∅`, `Predicted = {S2-00047}`)**: Penalized to $0.0$ ($\text{Precision} = 0.0, \text{Recall} = 1.0 \implies F_{0.5} = 0.0$).
- **Missed Match (`True = {S2-1}`, `Predicted = ∅`)**: Scores $0.0$ ($\text{Precision} = 0.0, \text{Recall} = 0.0 \implies F_{0.5} = 0.0$).
- **Exact Match (`True = P`, `Predicted = P`)**: Scores $1.0$.

### Country-Level Metric Infrastructure
Implemented in `src/evaluation/metrics.py`. Every evaluation call returns both:
1. `OVERALL`: Number of S1 entities, macro precision, macro recall, macro $F_{0.5}$.
2. `BY COUNTRY`: Per-country breakdown for every country string in the dataset (`US`, `India`, `France`, etc.).

---

## 12. Blocking Recall Infrastructure (Part F)

Implemented in `src/evaluation/measure_blocking_recall.py`.

### Key Capabilities
- Evaluates candidate generators against ground-truth mappings.
- Computes macro blocking recall across entities with $\ge 1$ true match.
- Handles 0-match entities separately without artificially inflating or deflating recall.
- Computes micro recall, count of entities with any missed match, count of entities with all missed matches, candidate count statistics (avg, median, max), and reduction ratio.
- Supports single blocking channel, multiple channels, and the **UNION** of multiple blocking channels.

---

## 13. Questions to Investigate in Future Normalization & Blocking Stages
1. **Transliteration Mapping**: How can script-normalization (e.g. converting Telugu/Devanagari/Kannada script tokens or ignoring non-ASCII script tokens when Latin script tokens match) boost candidate recall for Indian entities?
2. **Legal Suffix Stripping**: Which legal suffix tokens (`Pvt Ltd`, `Inc`, `LLP`, `P.C.`, `LLC`, `Corp`) should be normalized during blocking without losing discriminative power?
3. **Prefix & Honorific Cleaning**: What regex rules safely remove leading noise (`***`, `Sri`, `Smt`)?
4. **State Name Mapping**: How to construct a bidirectional mapping table for US states (`IN` $\leftrightarrow$ `Indiana`, `MN` $\leftrightarrow$ `Minnesota`) and Indian states (`TS` $\leftrightarrow$ `Telangana`, `UP` $\leftrightarrow$ `Uttar Pradesh`)?
5. **Multi-Channel Blocking Strategy**: What combination of exact token blocking, n-gram blocking, and min-hash / locality-sensitive indexing maximizes macro blocking recall while maintaining a high reduction ratio ($>99.9\%$) on 24 million records?

---

## 14. Verification & Unit Test Suite Summary
Unit test suite created in `tests/` covering:
- TSV loading & schema validation (`tests/test_loader.py`)
- $F_{0.5}$ metric edge cases, singletons, partial matches, country-level reporting (`tests/test_metrics.py`)
- Blocking recall evaluation, zero-match entities, channel union (`tests/test_blocking_recall.py`)

**Test Results**: `14 / 14` tests PASSED clean.
