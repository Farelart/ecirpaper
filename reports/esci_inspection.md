# Amazon ESCI: initial dataset inspection

Computed from the downloaded, checksum-verified official files.

Source revision and file checksums: `data/esci_manifest.json`.

Only training queries are shown below. Test rows are used only for structural statistics.

## Files and schemas

### examples: 2,621,288 rows

| column_name | column_type |
| --- | --- |
| example_id | BIGINT |
| query | VARCHAR |
| query_id | BIGINT |
| product_id | VARCHAR |
| product_locale | VARCHAR |
| esci_label | VARCHAR |
| small_version | BIGINT |
| large_version | BIGINT |
| split | VARCHAR |
| __index_level_0__ | BIGINT |

### products: 1,814,924 rows

| column_name | column_type |
| --- | --- |
| product_id | VARCHAR |
| product_title | VARCHAR |
| product_description | VARCHAR |
| product_bullet_point | VARCHAR |
| product_brand | VARCHAR |
| product_color | VARCHAR |
| product_locale | VARCHAR |

### sources: 130,652 rows

| column_name | column_type |
| --- | --- |
| query_id | BIGINT |
| source | VARCHAR |

## Example Splits

| product_locale | split | judgments | queries | products |
| --- | --- | --- | --- | --- |
| es | test | 93347 | 3844 | 80949 |
| es | train | 263063 | 11336 | 201686 |
| jp | test | 118907 | 4667 | 105906 |
| jp | train | 327146 | 13460 | 260320 |
| us | test | 425762 | 22458 | 360873 |
| us | train | 1393063 | 74888 | 982641 |

## Version Flags

| small_version | large_version | judgments |
| --- | --- | --- |
| 0 | 1 | 1503277 |
| 1 | 1 | 1118011 |

## Small Splits

| product_locale | split | judgments | queries | products |
| --- | --- | --- | --- | --- |
| es | test | 65883 | 2417 | 58122 |
| es | train | 152891 | 5632 | 123147 |
| jp | test | 88789 | 3123 | 79957 |
| jp | train | 209094 | 7284 | 171436 |
| us | test | 181701 | 8956 | 164900 |
| us | train | 419653 | 20888 | 351961 |

## Label Counts

| product_locale | split | esci_label | judgments |
| --- | --- | --- | --- |
| es | test | C | 4902 |
| es | test | E | 27074 |
| es | test | I | 11449 |
| es | test | S | 22458 |
| es | train | C | 12488 |
| es | train | E | 63131 |
| es | train | I | 26472 |
| es | train | S | 50800 |
| jp | test | C | 4033 |
| jp | test | E | 41597 |
| jp | test | I | 13604 |
| jp | test | S | 29555 |
| jp | train | C | 8970 |
| jp | train | E | 96170 |
| jp | train | I | 34454 |
| jp | train | S | 69500 |
| us | test | C | 8099 |
| us | test | E | 79708 |
| us | test | I | 30331 |
| us | test | S | 63563 |
| us | train | C | 19090 |
| us | train | E | 181819 |
| us | train | I | 71116 |
| us | train | S | 147628 |

## Candidates Per Query

| product_locale | split | min_candidates | median_candidates | mean_candidates | max_candidates | queries_with_duplicate_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| es | test | 8 | 23.000 | 27.258 | 61 | 0 |
| es | train | 8 | 21.000 | 27.147 | 77 | 0 |
| jp | test | 8 | 31.000 | 28.431 | 86 | 0 |
| jp | train | 8 | 35.000 | 28.706 | 96 | 0 |
| us | test | 8 | 16.000 | 20.288 | 95 | 0 |
| us | train | 8 | 16.000 | 20.091 | 188 | 0 |

## Product Fields

| product_locale | products | unique_ids | product_title_missing | product_title_mean_characters | product_description_missing | product_description_mean_characters | product_bullet_point_missing | product_bullet_point_mean_characters | product_brand_missing | product_brand_mean_characters | product_color_missing | product_color_mean_characters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| es | 260011 | 260011 | 0 | 115.831 | 118370 | 519.705 | 36977 | 596.338 | 12020 | 8.115 | 100088 | 5.904 |
| jp | 339059 | 339059 | 0 | 61.947 | 189640 | 154.466 | 88367 | 192.080 | 58401 | 8.216 | 184828 | 2.684 |
| us | 1215854 | 1215854 | 0 | 105.703 | 569795 | 446.453 | 178776 | 576.890 | 72335 | 9.199 | 406104 | 6.293 |

## Integrity

| duplicate_product_locale_keys | query_ids_in_multiple_splits | unmatched_judgments | invalid_labels |
| --- | --- | --- | --- |
| 0 | 1 | 0 | 0 |

## Small Split Integrity

| overlapping_query_ids |
| --- |
| 0 |

## Cross Split Ids All Versions

| query_id | product_locale | splits |
| --- | --- | --- |
| 79706 | us | 2 |

## Us Small Product Fields

| products | missing_description | missing_bullet_points | missing_description_percent |
| --- | --- | --- | --- |
| 482105 | 230347 | 63627 | 47.779 |

## Us Small Product Overlap

| union_products | train_products | test_products | shared_products |
| --- | --- | --- | --- |
| 482105 | 351961 | 164900 | 34756 |

## Us Small Query Lengths

| split | queries | median_words | mean_words | queries_at_least_six_words |
| --- | --- | --- | --- | --- |
| test | 8956 | 4.000 | 3.887 | 1245 |
| train | 20888 | 4.000 | 3.889 | 2959 |

## Sources

| source | queries |
| --- | --- |
| behavioral | 3780 |
| negations | 6964 |
| nlqec | 169 |
| other | 113656 |
| parse_pattern | 6083 |

## Training Samples

| query_id | query | esci_label | product_id | product_title |
| --- | --- | --- | --- | --- |
| 733 | 1 fl oz hand sanitizer bulk spread love not germa | C | B08DM6WQB3 | Wedding Favor LABELS - 20 Personalized Mini Hand Sanitizer STICKERS, Share Love Not Germs |
| 733 | 1 fl oz hand sanitizer bulk spread love not germa | E | B004T0GJDO | PURELL Advanced Hand Sanitizer Refreshing Gel, Clean Scent, 1 Fl Oz Travel Size Flip-Cap Bottle with Display Bowl (Pack of 36) – 3901-36-BWL |
| 733 | 1 fl oz hand sanitizer bulk spread love not germa | I | B06Y56QLJB | Mini Clear Plastic Spray Bottle Portable Cute Perfume Mouthwash Atomizer for Cleaning, Travel, Essential Oils, Perfume (60, 5ml) |
| 733 | 1 fl oz hand sanitizer bulk spread love not germa | S | B007IY3JE0 | Bath & Body Works Japanese Cherry Blossom 5-Pack PocketBac Sanitizers Anti-Bacterial Hand Gel 1 Fl Oz |
| 5117 | 7 x 5 acrylic sign holder landscape | E | B00NPBV528 | Dazzling Displays 7 x 5 Slanted Sign Holders (6-Pack) |
| 5117 | 7 x 5 acrylic sign holder landscape | S | B07HQLHLZC | MaxGear Wall Mount Sign Holder, 8.5x11 inches Acrylic Sign Holder with Mounting Screws, Plastic Sign Holder Clear Acrylic Picture Frame for Home, Office, Store, Restaurant, Landscape (12 Pack) |
| 24328 | center punch | E | B00004T7RH | General Tools 79 Steel Automatic Center Punch , Black |
| 24328 | center punch | S | B000E64RBI | Starrett S117PC Center Punch Set, Set of 5 - 1/16", 5/64", 3/32", 1/8", 5/32" Diameters, Hardened Tempered Steel Metal, Centering Tool for Machinists, Carpenters with Knurled Finger Grip, Storage Pouch |
| 75921 | orang comforters | E | B08B4SRBHF | Sheet Set 4 Piece Duvet Cover Set- Twin Size Halloween Trick or Treat Orang Black Ultra Soft Microfiber Quilt Cover with Zipper Closure (1 Comforter Cover + 1 Flat Sheet + 2 Pillow Shams) |
| 75921 | orang comforters | S | B076Q6KXWX | Softta Paisley Bedding Design 800 Thread Count 100% Cotton 3Pcs Duvet Cover Set,Cal King Size,Khaki |

## Sample Product Metadata

| product_id | product_title | product_brand | product_color | description_excerpt | bullet_point_excerpt |
| --- | --- | --- | --- | --- | --- |
| B00004T7RH | General Tools 79 Steel Automatic Center Punch , Black | General Tools | Black | None | DELIVERS A HIGH IMPACT STRIKE - one-handed spring-loaded impact operation, no hammer required ADJUSTABLE SPRING TENSION - adjustable spring tension cap for light or heavy strokes and marking ALL STEEL 5" BODY - center punch has an all-steel body, knurled finger grip and rust-resistant black oxide fi |
| B004T0GJDO | PURELL Advanced Hand Sanitizer Refreshing Gel, Clean Scent, 1 Fl Oz Travel Size Flip-Cap Bottle with Display Bowl (Pack of 36) – 3901-36-BWL | Purell | Clear | None | Show you care about the well-being of your team and visitors in the office and on-the-go Place PURELL hand sanitizer at desks, in common areas, and restrooms Perfect for frequent use. Mild formula with 4 moisturizers is free of triclosan, parabens, phthalates, and preservatives; so it does not dry o |
| B00NPBV528 | Dazzling Displays 7 x 5 Slanted Sign Holders (6-Pack) | Dazzling Displays | Clear | <b>Dazzling Displays 6-Pack of Landscape 7"W x 5"H Slanted Sign Holders</b> <p>This package of landscape 7"W x 5"H slanted sign holders is ideal for trade shows, retail stores, offices, presentations, counter PoP sales, and even in open houses in real-estate sales. </p> <p>Great for displaying pictu | SELF-STANDING. Self-standing and innovatively designed, our Dazzling Displays Sign Holders are ideal for photos, menus, advertising, drink specials and much more! It displays pictures and posters efficiently, and stands on its own without difficulty thanks to its wide, sturdy base. MULTIPURPOSE. Our |
| B08B4SRBHF | Sheet Set 4 Piece Duvet Cover Set- Twin Size Halloween Trick or Treat Orang Black Ultra Soft Microfiber Quilt Cover with Zipper Closure (1 Comforter Cover + 1 Flat Sheet + 2 Pillow Shams) | Millxiu | Halloween-021mix0158 | <b> Millxiu Luxury Ultra Soft High Quality Duvet Cover Sets Give You All Year Comfortable</b><br><br> <b>Soft Breathable Durable</b><br> <b>100% MICROFIBER POLYESTER</b><br> Our Bedding Set are made of microfiber polyester, which is soft, breathable and not easy to shrink.<br> Best gift for children | Breathable & Durable ：Millxiu's lightweight microfiber duvet cover sets are made of high quality polyester, that is soft and comfortable, keep you sleep well all night. Zipper Closure & Corner Ties：Each duvet cover has a zipper closure, concealed 4 corner ties help hold the duvet in place to ensure  |

## Implications for the experiment

- Identify products by `(product_id, product_locale)` and preserve official query splits.
- Begin with English (US), `small_version=1`, using the supplied candidate sets.
- Candidate judgments are not exhaustive whole-catalog judgments.
- Generate document expansions from product metadata only, excluding queries and relevance labels.
- Build development data from training query IDs and keep the official test split for final comparisons.
- Fix the same original product fields, tokenizer, BM25 settings, and relevance gains for all variants.
- Use fixed, complete candidate judgments for normalization; never build qrels from retrieved hits.
- Product overlap between train and test does not imply query leakage: document indexing is unsupervised.
- Missing descriptions affect the enrichment hypothesis and should be included in subgroup analysis.
- The full release has one cross-split query ID (79706, US); the small ranking subset has no overlapping query IDs.
- Candidate counts in the files can exceed the README's stated 40; use the actual supplied candidate lists.
- Verify the relevance-gain convention before implementing evaluation: the archived upstream conversion script's substitute/complement mapping deserves reconciliation with the benchmark paper.
- These data do not establish multi-turn conversational search effectiveness.
